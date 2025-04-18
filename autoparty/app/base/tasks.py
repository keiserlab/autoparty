"""
Functions to deal with loading molecules to database, generating fingerprints, generating 2d visualizations
"""
import os
import os.path

import json
import numpy as np
import pandas as pd
import datetime
import csv
import time
import glob
from zipfile import ZipFile

from sqlalchemy import desc
from sqlalchemy.sql import exists
from sqlalchemy.orm.exc import NoResultFound

from flask import current_app

from app import celery, db
from app.base.database import Preloaded, Molecule, Grade, Prediction, HPRunSetting
from app.base.luna_utils import run_mol, run_mol_batch, get_protein_cache
from app.base.hp_utils import _still_running, load_last_model
from app.base.io_utils import parse_party_config
from app.base.models import get_model
from app.base.molecule_utils import get_grades_for_training, get_molecules_for_predicting, get_predictions
from app.base.defaults import *

from celery import shared_task, Task
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import task_postrun
from celery.utils.log import get_task_logger

import rdkit.Chem as Chem

import torch
from torch import nn

import copy
import shutil

import pickle

logger = get_task_logger(__name__)

# Not celery task, but I'm putting here since it's used by celery tasks and it avoids circular dependencies
# Should probably go in molecules or a db_utils file technically, might move. 
def save_molecule(mol_name, run_id, mol, score, inter_jsons, ifp):

    mol = Molecule(name=mol_name,
                        run_id=run_id,
                        smi= Chem.MolToSmiles(mol),
                        score=score,
                        inters=json.dumps(inter_jsons),
                        mol= Chem.MolToMolBlock(mol),
                        ifp=ifp)

    db.session.add(mol)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()


# similar to above function, called by tasks but not strictly a task itself. Going here anyway
# to avoid circular dependencies
def save_prediction(mol_id, user_id, party_id, prediction, var, error):
    
    p = Prediction(mol_id = mol_id,
                    user_id = user_id,
                    hp_settings_id = party_id,
                    prediction = prediction,
                    uncertainty = var,
                    error=error)
    db.session.add(p)
    try:
        db.session.commit()
    except:
        db.session.rollback()

def apply_annotations(annotation_df, run_id, user_id, party_id, 
    grade_col = "grade", metacols = None):


    celery_task_list = []

    molecules = Molecule.query.filter_by(run_id = run_id
        ).filter(Molecule.name.in_(set(annotation_df.index))).all()

    for mol in molecules:
        grade = annotation_df.at[mol.name, grade_col] if grade_col in annotation_df.columns else None
        metavals = annotation_df.loc[mol.name, metacols] if metacols is not None else None
        if grade is None and metacols is None:
            raise ValueError # both grades and metadata are None, nothing to add
        celery_task_list.append(task_save_grade_helper.apply(args=(mol.id, grade, run_id, user_id, party_id), 
            kwargs={ 'metacols': metacols, 'metavals': metavals}))
    
    return {'complete': sum(celery_task_list), 'total':annotation_df.shape[0]}

@celery.task(bind=True)
def task_calculate_interactions(self, mol_names, mol_strs, scores, run_id, runname, pdb_file, luna_config):
    """Calculate interactions using LUNA

    Args:
        :param str mol_name: Name of molecule, read in from molecule file
        :param str mol: String representation of molecule, generated with rdkit Chem.MolToMolBlock
        :param float score: Molecule score - virtual docking score, binding affinity, etc
        :param int run_id: Screen id in database, used to filter molecules during hitpicking party
        :param str pdb_file: Path to PDB file for interaction calculation
        :param str luna_config: Path to LUNA overall config file, used for interaction calculation 

    """
    try:
        self.update_state(state="PROGRESS",
            meta = {'complete': 0, 'total':len(mol_names)})

        luna_config = json.loads(luna_config)
        mols = [Chem.MolFromMolBlock(mol) for mol in mol_strs]

        #cache protein features
        cache, params = get_protein_cache(runname, mol_names[0], mols[0], pdb_file)

        for j, (mol_name, mol, score) in enumerate(zip(mol_names, mols, scores)):
            ifp, inter = run_mol(runname, mol_name, mol, pdb_file, 
                                                            params = params,
                                                            cache = cache)
            inter_jsons = [i.as_json() for i in inter.interactions]
            save_molecule(mol_name, run_id, mol, score, inter_jsons, ifp)
            self.update_state(state = "PROGRESS", meta = {"complete": j+1, "total": len(mols)})

        return {'complete': j+1, 'total':len(mol_names)} 
    except Exception as e:
        logger.INFO(f"TASK FAILED: {e}")
        return {'processed':False, '':e}

def task_save_grade_helper(mol_id, grade, run_id, user_id, party_id, 
    metacols = None, metavals = None, build_ifp = True, dummy_ifp = None):

    try:
        #update molecular metadata, if provided
        if metacols is not None:
            Molecule.query.filter_by(id = mol_id).update({Molecule.meta: json.dumps(dict([(key, val) for key, val in zip(metacols, metavals)]))})

        if not pd.isnull(grade):
            try: # check if grade exists
                existing_grade = Grade.query.filter_by(mol_id = mol_id
                    ).filter_by(hp_settings_id = party_id
                    ).filter_by(user_id = user_id).one()
                Grade.query.filter_by(mol_id = mol_id
                    ).filter_by(hp_settings_id = party_id
                    ).filter_by(user_id = user_id
                    ).update({Grade.grade: grade, Grade.timestamp: datetime.datetime.now()})

            except NoResultFound: # create new grade
                # create fingerprint to use for model training later
                if build_ifp:
                    mol = Molecule.query.get(mol_id)
                    counts = {int(index):int(count) for (index, count) in mol.ifp.counts.items()}
                else:
                    counts = dummy_ifp

                grade = Grade(
                            run_id = run_id,
                            user_id = user_id,
                            hp_settings_id = party_id,
                            mol_id=mol_id,
                            ifp=json.dumps(counts),
                            grade=grade)

                db.session.add(grade)
        
            db.session.commit()
        return True
    except Exception as e:
        return False

@celery.task(bind = True)
def task_train_model(self, run_id, user_id, hp_settings_id, run_val = True, np_seed = 27):
    try:

        party_config, errors = parse_party_config(dry_run = True)# load default settings
        party_config.update(json.loads(HPRunSetting.query.get(hp_settings_id).party_config))

        logger.info(party_config)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(party_config, dev = dev)

        grade_df = get_grades_for_training(hp_settings_id)
        
        training_all = []
        validation_all = []
        epoch = 0
        self.update_state( state = "TRAINING",
                            meta = {'num_grades': grade_df.shape[0],
                                    'training': training_all,
                                    'validation': validation_all,
                                    'epoch': epoch})

        # TODO: Update this number + up max_epochs
        #Currently always runs with validation if there are at least 20 annotations - change?
        run_val = run_val and grade_df.shape[0] > 50

        valloader = None
        if run_val:
            num_val = int(grade_df.index.shape[0]/10)
            val_ids = np.random.choice(list(grade_df.index), num_val, replace = False)
            trainloader, valloader = model.create_dataloaders(grade_df.drop(val_ids), grade_df.loc[val_ids])
        else:
            trainloader = model.create_dataloaders(grade_df)

        best_epoch = -1; best_val = float("inf"); patience = party_config['patience']

        model_filename = "-".join([f"{val_id}{val}"for val_id, val in zip(['u', 's', 'p'], [user_id, run_id, hp_settings_id])]) + \
            f"_{self.request.id}"

        for epoch in range(party_config['max_epochs']):
            training, validation = model.train_single_epoch(trainloader, valloader)
            training_all.append(training)

            if validation > 0: # validation > 0 iff run_val is true, otherwise -1
                validation_all.append(validation)

            if len(training_all[0]) > 0:
                train_return = list(map(list, zip(*training_all)))
            else:
                train_return = training_all


            if run_val:
                if validation < best_val:
                    best_val = validation
                    best_epoch = epoch
                    model.save_model(f"{UPLOAD_FOLDER}/{model_filename}") # save model
                    patience_count = 0
                else:
                    patience_count += 1

            else:
                best_epoch = best_epoch

            self.update_state(state="TRAINING",
                                meta = {'num_grades': grade_df.shape[0],
                                    'training': train_return, 
                                    'validation': validation_all,
                                    'epoch': epoch,
                                    'best_epoch': best_epoch})

            if run_val:
                if patience_count > patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        if not run_val:
            # save at the end since we didn't save it earlier during validation
            model.save_model(f"{UPLOAD_FOLDER}/{model_filename}")

        np.save(f"{UPLOAD_FOLDER}/{model_filename}_training", np.array(train_return))
        min_val = -1
        if run_val:
            np.save(f"{UPLOAD_FOLDER}/{model_filename}_validation", np.array(validation_all))
            min_val = min(validation_all)

        return {"success":True, "train_loss": min(np.mean(train_return, axis=1)), "val_loss": min_val, 'train_size': grade_df.shape[0]}

    except Exception as e:
        return {"success": False, "error": str(e)}

@celery.task(bind = True)
def task_predict_molecules(self, mol_ids, run_id, user_id, hp_settings_id, train_task_id):
    
    self.update_state(state="PROGRESS",
            meta = {'complete': 0, 'total':len(mol_ids)})

    # load model from saved state - need LUNA config
    party_config, errors = parse_party_config(dry_run = True)# load default settings
    party_config.update(json.loads(HPRunSetting.query.get(hp_settings_id).party_config))

    error_dict = None
    if party_config['output_type'] == "ordinal":
        error_dict = {}
        for i, option in enumerate(party_config['output_options']):
            error_dict[option] = i
    
    model_filename = "-".join([f"{val_id}{val}"for val_id, val in zip(['u', 's', 'p'], [user_id, run_id, hp_settings_id])]) + \
            f"_{train_task_id}" 
    
    dev = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(party_config)
    model.load_from_state(f"{UPLOAD_FOLDER}/{model_filename}")

    # get molecules by id
    molecule_df = get_molecules_for_predicting(mol_ids)
    predictloader = model.create_dataloaders(molecule_df, predict = True, batch_size = len(mol_ids))

    preds, uncerts = model.predict(next(iter(predictloader)).to(dev))

    preds = predictloader.dataset.reverse_convert(preds)

    # get actual grades to calculate error
    grade_df = get_grades_for_training(hp_settings_id)

    for i, (mol_id, pred, uncert) in enumerate(zip(mol_ids, preds, uncerts)):

        try:
            grade = grade_df.at[mol_id, "label"]
            if error_dict is not None:
                error = abs(error_dict[grade] - error_dict[pred])
            else:
                error = 1 # flat 1 if incorrect

        except:
            grade = None # many molecules will not have grades 
            error = None

        try:
            # update existing prediction

            existing_pred = Prediction.query.filter_by(mol_id = mol_id
                ).filter_by(hp_settings_id = hp_settings_id).one_or_none()

            if existing_pred is None:
                p = Prediction(mol_id = mol_id,
                       user_id = user_id,
                       hp_settings_id = hp_settings_id,
                       prediction = pred,
                       uncertainty = uncert,
                       error = error)

                db.session.add(p)
            else:
                Prediction.query.filter_by(mol_id = mol_id
                    ).filter_by(hp_settings_id = hp_settings_id
                    ).update({Prediction.prediction: pred, Prediction.uncertainty: uncert, Prediction.error: error})

            self.update_state(state="PROGRESS",
                      meta = {'complete':i + 1, 'total':len(mol_ids)})

        except: # prediction doesn't exist yet, make a new one
            pass

    try:
        db.session.commit() # wait until all are updated, otherwise this leads to rapid reordering for the user
    except:
        db.session.rollback()
    return {'complete':i + 1, 'total':len(mol_ids)}

@task_postrun.connect
def close_session(*args, **kwargs):
    db.session.remove()
    
    
    
    
    
       
