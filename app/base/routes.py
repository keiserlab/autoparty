from flask import (
    jsonify, 
    render_template, 
    redirect, 
    request, 
    url_for, 
    session, 
    g, 
    flash, 
    send_file,
    current_app,
    make_response,
    send_from_directory
)
from flask_login import (
    current_user,
    login_required,
    login_user,
    logout_user
)

import sys
import os
import numpy as np
import math

from sqlalchemy import desc
from sqlalchemy.sql import exists
from sqlalchemy.orm.exc import NoResultFound

import multiprocessing
import tqdm

import pandas as pd
import json
import random
import datetime
from datetime import timezone

import pickle

from celery import group 

import torch
from torch import nn

import openbabel.pybel
openbabel.pybel.ob.obErrorLog.StopLogging()

import rdkit
from rdkit import Chem

from collections import deque

import gzip
from zipfile import ZipFile
import glob
import io

from app import celery, db, login_manager
from app.base import blueprint
from app.base.forms import LoginForm, CreateAccountForm, ScreenForm, GradeForm, HitpickerRunForm, MethodForm
from app.base.database import User, Preloaded, Molecule, Grade, HPRunSetting, Prediction
from app.base.util import verify_pass, istarmap
from app.base.tasks import task_calculate_interactions, task_train_model, task_save_grade_helper, apply_annotations, task_predict_molecules
from app.base.hp_utils import populate_queue, _dump_q_to_list
from app.base.io_utils import check_remote_file, load_remote_molecules, load_remote_pdb_file, load_existing_annotations, check_luna_config, parse_party_config, remove_files, recover_curves, prepare_files
from app.base.molecule_utils import get_molecule_by_id
from app.base.molecule_utils import get_num_grades, prediction_csv
from app.base.molecule_utils import update_history, recover_history
from app.base.molecule_utils import get_ordered_molecules

from multiprocessing import Pool

from app.base.defaults import *

@blueprint.route('/')
def route_default():
    return redirect(url_for('base_blueprint.run_settings'))

## Login & Registration
@login_manager.user_loader
def load_user(user_id):
    return User.query.filter_by(id=str(user_id)).first()


@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    login_form = LoginForm(request.form)
    if 'login' in request.form:
        
        # read form data
        username = request.form['username']
        password = request.form['password']

        # Locate user
        user = User.query.filter_by(username=username).first()
        
        # Check the password
        if user and verify_pass( password, user.password):
            
            login_user(user)
            session['user_id'] = user.id

            return redirect(url_for('base_blueprint.route_default'))

        # Something (user or pass) is not ok
        return render_template( 'accounts/login.html', msg='Wrong user or password', form=login_form)

    if not current_user.is_authenticated:
        return render_template( 'accounts/login.html',
                                form=login_form)
    return redirect(url_for('home_blueprint.index'))

@blueprint.route('/register', methods=['GET', 'POST'])
def register():
    login_form = LoginForm(request.form)
    create_account_form = CreateAccountForm(request.form)

    if 'register' in request.form:
        username  = request.form['username']

        # Check usename exists
        user = User.query.filter_by(username=username).first()
        if user:
            return render_template( 'accounts/register.html', 
                                    msg='Username already registered',
                                    success=False,
                                    form=create_account_form)


        # else we can create the user
        user = User(**request.form)
        db.session.add(user)
        db.session.commit()
        
        user = User.query.filter_by(username=username).first()
        session['user_id'] = user.id
        login_user(user)
        return redirect(url_for('base_blueprint.route_default'))

    else:
        return render_template( 'accounts/register.html', form=create_account_form)
 
@blueprint.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('base_blueprint.login'))

@blueprint.route('/load-remote', methods = ['POST'])
def load_local():

    newscreen = int(request.form["prevscreen"]) == 0

    # check new screen things
    newscreen_errors = []
    if not newscreen and request.form['runname'] != "":
        newscreen_errors.append("Cannot provide name if appending to existing screen.")
    
    if not newscreen and request.form['recpdb'] != "":
        newscreen_errors.append("Cannot provide name if appending to existing screen.")
    if not newscreen and len(request.files.getlist('lunaconfig')) > 0:
        newscreen_errors.append("Cannot provide LUNA configuration files if appending to existin screen.")
    if newscreen_errors:    
        return jsonify({'processed':False, 'reasons':newscreen_errors})

    req_files = ['mollist']
    if newscreen:
        req_files += ['recpdb']

    # do the files exist
    files_ok = dict([(file_id, check_remote_file(request.form[file_id])) for file_id in req_files])
    if not all(files_ok.values()): return jsonify({'processed':False, 'reasons': ['One or more files not found.']})

    # is the pdb file structured correctly
    if newscreen:
        pdbcontents = load_remote_pdb_file(request.form['recpdb'], request.form['runname'])
        if not pdbcontents: files_ok['recpdb'] = False; return jsonify({'processed':False, 'reasons':['Invalid PDB file.']})

    # try to load mols from provided file
    mols, mol_errors = load_remote_molecules(request.form['mollist'], request.form['scorelabel']) 
    if mol_errors: return jsonify({'processed': False, 'reasons': mol_errors})

    if newscreen:

        # add screen to database
        new_screen = Preloaded(runname=request.form['runname'], 
                            pdbcontents=pdbcontents, 
                            user_id=current_user.get_id())
        db.session.add(new_screen)
        db.session.commit()

        pdbpath = request.form['recpdb']

        screen_id = new_screen.id # get id here, used to name luna config files in uploads folder

        # validate LUNA settings
        luna_config, luna_errors = check_luna_config(request.files.getlist('lunaconfig'), user_id = int(current_user.get_id()), screen_id = screen_id)
        if luna_errors:  # slightly more error handling required, need to rollback the screen from the database
            db.session.query(Preloaded).filter_by(id = screen_id).delete()
            db.session.commit()
            return jsonify({'processed':False, 'reasons':luna_errors})

        # luna config processed, correcly, need to update database entry
        Preloaded.query.filter_by(id = screen_id).update({Preloaded.luna_config: json.dumps(luna_config)})
        db.session.commit()

        # save pdb contents in uploads folder in case we need it later
        upload_pdb_path = f"{UPLOAD_FOLDER}/u{current_user.get_id()}-s{screen_id}_protein.pdb"
        with open(upload_pdb_path, 'w+') as f:
            f.write(pdbcontents)

        runname = request.form['runname']

    else:
        # appending to existing screen, get screen, pdbpath, luna_config
        screen_id = request.form["prevscreen"]
        screen = Preloaded.query.filter_by(id = screen_id).one()
        upload_pdb_path = f"{UPLOAD_FOLDER}/u{current_user.get_id()}-s{screen_id}_protein.pdb"
        if not os.path.exists(upload_pdb_path):
            with open(upload_pdb_path, 'w+') as f:
                f.write(screen.pdbcontents)
        runname = screen.runname
        luna_config = json.loads(screen.luna_config)

    # start interaction calculations
    batch_size = MOL_BATCH_SIZE

    names, mols, scores = list(zip(*mols))
    mol_batches = [(names[i:i+batch_size], mols[i:i+batch_size], scores[i:i+batch_size]) 
                            for i in range(0, len(names), batch_size)]

    mol_calc_group = group(task_calculate_interactions.s(names, mols, scores, screen_id, 
                                    runname, upload_pdb_path, json.dumps(luna_config)) for names, mols, scores in mol_batches)
    mol_calc_result = mol_calc_group.apply_async()
    mol_calc_result.save()

    return jsonify({'processed':True, 'mol_calc_id': mol_calc_result.id})


@blueprint.route('/load-screen', methods = ['GET', 'POST'])
@login_required
def docking_screens():
    # only allow logged in users to upload content
    login_form = LoginForm(request.form)
    if not current_user.is_authenticated:
        return render_template( 'accounts/login.html', form=login_form)

    screen_form = ScreenForm()

    # get existing screen to populate dropdown
    runs = Preloaded.query.filter_by(user_id = current_user.get_id()
                            ).order_by(desc(Preloaded.timestamp)).all()
    screen_options = [(0, "New screen")]
    if len(runs) == 0:
        screen_options = [(0, "No screens found")]
    for run in runs:
        screen_options.append(( run.id, f'{run.runname} - created by {run.user} on ' +\
         f'{run.timestamp.strftime("%m/%d/%Y, %H:%M:%S")}'))

    screen_form.existing_screens.choices = screen_options

    return render_template('load-screen.html', form = screen_form)


@blueprint.route('/run-settings', methods = ['GET', 'POST'])
@login_required
def run_settings():

    login_form = LoginForm(request.form)
    if not current_user.is_authenticated:
        return render_template( 'accounts/login.html',
                                form=login_form)
    
    if request.method == "POST":
        
        session['mol_queue'] = []

        errors = []
        
        if "new-party" in request.form:
            # create a new party
            if int(request.form['preloaded_id']) <= 0:
                errors.append("No screens loaded - please load a screen first.")

            if errors:
                return jsonify({'processed': False, 'reasons': errors})

            else:
                new_hp_setting = HPRunSetting(**request.form, 
                        funnel = False,
                        user_id = current_user.get_id())
                db.session.add(new_hp_setting)
                db.session.commit()

                #set session variables
                session['hp_settings_id'] = new_hp_setting.id
                session['run_id'] = request.form['preloaded_id']

                # check party config
                party_config, config_errors = parse_party_config([request.files['party_config']],
                                                            user_id = int(current_user.get_id()),
                                                            screen_id = int(session['run_id']),
                                                            party_id = int(session['hp_settings_id']),
                                                            new_filename = "party.conf")
                errors = list(config_errors.values())

                if errors:
                    # rollback party
                    db.session.query(HPRunSetting).filter_by(id = session['hp_settings_id']).delete()
                    db.session.commit()
                    return jsonify({'processed': False, 'reasons': errors})
                else:
                    db.session.query(HPRunSetting).filter_by(id = session['hp_settings_id']
                        ).update({HPRunSetting.party_config : json.dumps(party_config)})
                    db.session.commit()
                return jsonify({'processed': True})


        else:
            # set session variables
            hp_settings_int = int(request.form['hp_settings_id'])
            hp_settings = HPRunSetting.query.get(int(request.form['hp_settings_id']))
            session['hp_settings_id'] = hp_settings.id
            session['run_id'] = hp_settings.preloaded_id

            # update timestamp
            HPRunSetting.query.filter_by(id = hp_settings.id
                ).update({HPRunSetting.timestamp:datetime.datetime.now()})
            db.session.commit()
            return jsonify({'processed': True})


    # populate list of existing runs
    runs = Preloaded.query.order_by(desc(Preloaded.timestamp)).all()
    screen_options = []
    if len(runs) == 0:
        screen_options = [(-1, "No loaded screens found")]
    for run in runs:
        screen_options.append(( run.id, f'{run.runname} - loaded by {run.user} on ' +\
         f'{run.timestamp.strftime("%m/%d/%Y, %H:%M:%S")}'))
    
    #populate list of existing parties for this user
    parties = HPRunSetting.query.filter_by(user_id = current_user.get_id()
        ).order_by(desc(HPRunSetting.timestamp)).all()
    party_options = []

    if parties:
        for party in parties:
            party_str = f'Screen: {party.preloaded.runname}' +\
                f' - last loaded {party.timestamp.strftime("%m/%d/%Y, %H:%M:%S")}'
            party_options.append((party.id, party_str))
    else:
        party_options = [(-1, "No existing parties found")]

    hp_run_form = HitpickerRunForm()
    hp_run_form.preloaded_id.choices = screen_options
    hp_run_form.hp_settings_id.choices = party_options

    return render_template('run-settings.html', form = hp_run_form)


@blueprint.route('/hp-in-progress', methods = ['GET'])
@login_required
def label_mols():
    
    login_form = LoginForm(request.form)
    if not current_user.is_authenticated:
        return render_template( 'accounts/login.html',
                                form=login_form)
    
    if 'run_id' not in session:
        return redirect(url_for('base_blueprint.docking_runs'))
    
    grade_form = GradeForm(request.form)
    method_form = MethodForm(request.form)

    # populate grade form with provided options
    settings = HPRunSetting.query.filter_by(id = session['hp_settings_id']).one()
    party_config = json.loads(settings.party_config)

    choices = []
    for option in party_config['output_options']:
        choices.append((option, str(option).upper()))

    grade_form.choice_switcher.choices = choices

    run = Preloaded.query.filter_by(id=session['run_id']).one()
    cur_pdb = run.pdbcontents

    return render_template('hitpicking_in_progress.html', 
                          pdb_file=cur_pdb,
                          gradeform=grade_form,
                          methodform=method_form)

@blueprint.route("/start-train")
def start_train():
    train_task = task_train_model.apply_async((session['run_id'], current_user.get_id(), session['hp_settings_id']))
    return jsonify({"processed": True, "task_id": train_task.id})

@blueprint.route("/start-predict")
def start_predict():
    train_task_id = request.args.get('task_id')

    # batching molecules for this too
    mol_ids = [mol.id for mol in Molecule.query.filter_by(run_id = session['run_id']).all()]
    batch_size = MOL_BATCH_SIZE

    mol_batches = [mol_ids[i:i+batch_size] for i in range(0, len(mol_ids), batch_size)]

    mol_pred_group = group(task_predict_molecules.s(mols, session['run_id'], current_user.get_id(), session['hp_settings_id'], train_task_id) 
                                for mols in mol_batches)
    
    mol_pred_result = mol_pred_group.apply_async()
    mol_pred_result.save()
    return jsonify({"processed": True, "pred_task_id": mol_pred_result.id})

@blueprint.route('/process-annotations', methods = ['POST'])
def process_annotations():
    # load grades

    # unfortunately a lot of the processing needs to be done here (issues with app context, passing dfs to celery tasks)
    # query relevant molecules


    annotations = request.files.getlist('annotations')
    grades, metacols = load_existing_annotations(annotations, user_id = int(current_user.get_id()), 
                                                              screen_id = int(session['run_id']),
                                                              party_id = int(session['hp_settings_id']))

    grades['grade'] = grades['grade'].apply(str) # convert all to strings
    grades['grade'] = grades['grade'].apply(lambda x: int(x) if x.isdigit() else x.lower())
    
    # check duplicates
    duplicated = list(grades[grades.index.duplicated()].index)
    grades = grades[~grades.index.duplicated(keep='first')]

    # check that grades are allowed according to party config
    party_config, errors = parse_party_config(dry_run = True)# load default settings
    party_config.update(json.loads(HPRunSetting.query.get(session['hp_settings_id']).party_config))

    illegal = grades.loc[~grades['grade'].isin(party_config['output_options'])].index
    results_list = []

    grades = grades.loc[grades['grade'].isin(party_config['output_options'])]

    molecules = Molecule.query.filter_by(run_id = session['run_id']
        ).filter(Molecule.name.in_(set(grades.index))).all()

    unused = set(grades.index)

    for mol in molecules:
        try:
            grade = grades.at[mol.name, "grade"]
            if len(grade) > 1:
                duplicates.append(mol.name)
                grade = grades[0]
        except Exception as e:
            grade = None

        if metacols is not None: metavals = grades.loc[mol.name, metacols]
        else: metavals = None

        if grade is None and metacols is None:
            continue
        
        unused.remove(mol.name)

        results_list.append(task_save_grade_helper(
            mol.id, grade, session['run_id'], current_user.get_id(), session['hp_settings_id'], 
            metacols = metacols, metavals =  metavals))


    return jsonify({'processed': sum(results_list) == len(results_list) 
                        and len(duplicated) == len(unused) == 0,
                    'total_annotations': len(results_list),
                    'success_annotations': sum(results_list),
                    'duplicates':list(set(duplicated)),
                    'unused':list(unused),
                    'illegal':list(illegal)});

@blueprint.route('/save-grade', methods = ['POST'])
def save_grade():
    """
    Saves mol and associated grade in database
    """
    mol_data = request.get_json()
    #session.pop(mol_data['id'])
    save_grade_task = task_save_grade_helper(mol_data['id'], mol_data['grade'], 
        session['run_id'], current_user.get_id(), session['hp_settings_id'])
    return jsonify(results = {'processed': save_grade_task})

@blueprint.route('/check-train', methods = ['GET'])
def check_train():
    """
    Decides if we need to train a new model
    Actual training handled seperately, since we need to return that result async
    """
    force = request.args.get('force') # first reason to train - requested by the user
    num_grades = get_num_grades(session['hp_settings_id']) # second reason to train - we hit the req # of grades
    retrain_freq = json.loads(HPRunSetting.query.filter_by(
        id = session['hp_settings_id']).one().party_config)["retrain_frequency"]

    return jsonify({'train':any([force, (num_grades %  retrain_freq) == 0])})

@blueprint.route('/taskstatus', methods = ['GET'])
def taskstatus():
    task_id = request.args.get('task_id')
    screen_id = request.args.get('screen_id')
    delete_id = request.args.get('delete_id')

    if task_id: # model training
        task = task_train_model.AsyncResult(task_id)
        results = {"processed": task.ready(), "state":task.state}

        if task.state == "TRAINING":
            results.update(task.info)

        elif delete_id: # we shouldn't be deleting if its still training
            # update party history with new model information - do we need to add accuracy?
            results = task.info
            task.forget() # good to delete

            # handle case where training fails gracefully
            if results['success']:
                # remove prior models
                prefix = f"{UPLOAD_FOLDER}/" + "-".join([f"{val_id}{val}"for val_id, val in zip(['u', 's', 'p'], 
                    [current_user.get_id(), session['run_id'], session['hp_settings_id']])])
                remove_files([task_id], prefix = prefix, suffix = ".pt", inverse = True) # delete model files for this run that don't contain this task id
                remove_files([task_id], prefix = prefix, suffix = ".npy", inverse = True)

                # update party history with new model information
                update_history(current_user.get_id(), session['run_id'], session['hp_settings_id'], results)                

        return jsonify(results)

    elif screen_id: # molecule uploads, prediction  
        # for all tasks - are we done?
        result = celery.GroupResult.restore(screen_id)
        tasks_failed = result.failed()

        total = 0; completed_count = 0
        for task in result:
            if task.state == "PROGRESS" or task.state == "SUCCESS":
                total += task.info.get('total')
                completed_count += task.info.get('complete')
            else:
                total += MOL_BATCH_SIZE # assume batch size if pending, since we can't actually get the number requested

        all_finished = len(result.results) == result.completed_count() or tasks_failed

        if delete_id:
            result.forget() # we're good to delete
        
        return jsonify({"finished": all_finished, "completed_count":completed_count, "total": total, "failed": tasks_failed})

@blueprint.route('/prep-results', methods = ['GET'])
def prep_results():
    # first - make predictions, nmot async cause we need it to finish before file download
    predictions = prediction_csv(current_user.get_id(), session['run_id'], session['hp_settings_id'])
    all_results = prepare_files(current_user.get_id(), session['run_id'], session['hp_settings_id'])
    history = recover_history(session['hp_settings_id'])
    return jsonify({'processed': predictions and all_results, 'history': history})

@blueprint.route('/recover-trained')
def recover_trained():
    """
    Get training/validation data to recover the loss curves from a saved.
    """
    training, validation = recover_curves(current_user.get_id(), session['run_id'], session['hp_settings_id'])
    if training is None:
        return jsonify({"processed": False})
    else:
        epoch = training.shape[1] - 1
        best_epoch = training.shape[1] - 1
        if validation is not None:
            best_epoch = np.argmin(validation)
            validation = validation.tolist()
        else:
            validation = [] #empty, don't plot

    history = recover_history(session['hp_settings_id'])

    return jsonify({"processed": True, "training": training.tolist(), "validation": validation,
         "epoch":int(epoch), "best_epoch":int(best_epoch), "history":history})

@blueprint.route('/update-mol-queue', methods = ['POST'])
def update_mol_queue(page, method, mode, name, modetime):

    offset = page*MOL_PAGE_SIZE

    mols, preds, grades = None, None, None

    if method != 'name':
        name = None
    mols, grades, preds, total = get_ordered_molecules(session['run_id'], session['hp_settings_id'],
            name = name, orderby = method, mode = mode, modetime = modetime,
            limit = MOL_PAGE_SIZE, offset = offset)

    if mols is not None: 
        mols = Molecule.serialize_list(mols) # serialize to allow us to pass back to frontend
    
    # adjust preds, grades for serializaion + play nice with frontend
    #if any([pred is not None for pred in preds]):
    preds = [Prediction.serialize(pred) if pred is not None else [] for pred in preds]
    #else:
    #    preds = None
    
    #if any([grade is not None for grade in grades]):
    grades = [Grade.serialize(grade) if grade is not None else [] for grade in grades]
    #else: grades = None

    session['num_pages'] = int(math.ceil(total / MOL_PAGE_SIZE))
    session['mol_queue'] = deque()

    for mol, pred, grade in zip(mols, preds, grades):
        session['mol_queue'].append((mol, pred, grade))

    return jsonify({"processed": True, "molecules": len(mols)})


@blueprint.route('/get-mols', methods = ['POST'])
def get_mols():
    """
    Returns molecules from prepopulated queue, leads to fewer DB queries and better user experience overall
    """

    page = request.get_json()['page']
    method = request.get_json()['method']
    mode = request.get_json()['mode']
    name = request.get_json()['name']

    modetime = datetime.datetime.strptime(request.get_json()['modetime'], 
                                                "%a, %d %b %Y %H:%M:%S %Z")
    # convert utc to local time, since thats the timestamp grades are saved with
    modetime = modetime.replace(tzinfo=timezone.utc).astimezone(tz=None)

    offset = page*MOL_PAGE_SIZE

    mols = []
    preds = []
    grades = []

    if len(session['mol_queue']) == 0:
        offset = page*MOL_PAGE_SIZE

    mols, preds, grades = None, None, None

    if method != 'name':
        name = None
    mols, grades, preds, total = get_ordered_molecules(session['run_id'], session['hp_settings_id'],
            name = name, orderby = method, mode = mode, modetime = modetime,
            limit = MOL_PAGE_SIZE, offset = offset)

    if mols is not None: 
        mols = Molecule.serialize_list(mols) # serialize to allow us to pass back to frontend
    
    # adjust preds, grades for serializaion + play nice with frontend
    #if any([pred is not None for pred in preds]):
    preds = [Prediction.serialize(pred) if pred is not None else [] for pred in preds]
    #else:
    #    preds = None
    
    #if any([grade is not None for grade in grades]):
    grades = [Grade.serialize(grade) if grade is not None else [] for grade in grades]
    #else: grades = None

    session['num_pages'] = int(math.ceil(total / MOL_PAGE_SIZE))

    for mol, pred, grade in session['mol_queue']:
        mols.append(mol); preds.append(pred); grades.append(grade)
    
    # serialize
    #mols = Molecule.serialize_list(mols) # serialize to allow us to pass back to frontend
    #preds = [Prediction.serialize(pred) if pred is not None else [] for pred in preds]
    #grades = [Grade.serialize(grade) if grade is not None else [] for grade in grades] 

    refresh = len(session['mol_queue']) < MIN_MOL_QUEUE_SIZE

    return jsonify({'processed': mols is not None and len(mols) > 0, 'molecules': mols, 'predictions': preds, 
        'grades':grades, 'num_pages':session['num_pages'], 'refresh':refresh})

@blueprint.route('/get-mol', methods = ['POST'])
def get_mol():
    """
    Gets molecule and associated grade and prediction, if they exist, by id
    """
    mol_id = request.get_json()['id']

    try:
        # check if molecule is in the queue
        next_mol, next_pred, next_grade = session[mol_id]
    except:
        # fetch if not
        next_mol, next_pred, next_grade = get_molecule_by_id(request.get_json()['id'], 
            return_pred = True, return_grade = True,
            party_id = session['hp_settings_id'])
        #

    next_mol = Molecule.serialize(next_mol)
    if next_pred:
        next_pred = Prediction.serialize(next_pred)
    if next_grade:
        next_grade = Grade.serialize(next_grade)

    return jsonify({'processed' : True, 
                    'mol': next_mol,
                    'pred': next_pred,
                    'grade': next_grade})


@blueprint.route('/download')
def download():
    """
    Filetype: which files to download
    """
    filename = request.args.get('filename')
    extension = request.args.get('extension')
    append = request.args.get('append')

    filedir = DEFAULT_FOLDER

    if append:
        prefix = "-".join([f"{val_id}{val}"for val_id, val in 
            zip(['u', 's', 'p'], [current_user.get_id(), session['run_id'], session['hp_settings_id']])])
        filename = f'{prefix}_{filename}'
        filedir = UPLOAD_FOLDER

    if extension:
        filename += "." + extension

    output_dir = os.path.join(os.getcwd(), filedir)

    try:
        return send_from_directory(output_dir, filename, as_attachment = True)
    except:
        return jsonify({"processed": False, "reason": "Invalid file request"})

@blueprint.route("/information")
def autoparty_info():
    info = ""
    return render_template('autoparty-info.html', info = info)

## Errors
@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('page-403.html'), 403

@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('page-403.html'), 403

@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('page-404.html'), 404

@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('page-500.html'), 500
