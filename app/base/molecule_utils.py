
import pandas as pd

import os
import glob
import zipfile
import datetime

import json
import numpy as np

from sqlalchemy import desc, and_, or_
from sqlalchemy.sql import exists, false
from sqlalchemy.orm.exc import NoResultFound

from app import celery, db
from app.base.database import Molecule, Grade, Prediction, HPRunSetting, Model
from app.base.defaults import *

orderby_dict = {"score": [Molecule.score], "uncertainty": [desc(Prediction.uncertainty)], 
				"prediction": [Prediction.prediction, Prediction.uncertainty], 
				"disagreement": [desc(Prediction.error)]}

# no reason for these to be tasks really, they're synchronous
def get_molecule_by_id(mol_id, return_grade = False, return_pred = False, party_id = -1):
	"""
	Returns single molecule by id and grade if requested and available.
	"""
	mol = Molecule.query.get(mol_id)

	if return_grade:
		try:
			grade = Grade.query.filter_by(mol_id = mol_id
				).filter_by(hp_settings_id = party_id).one()
		except:
			grade = None
		if not return_pred:
			return mol, grade
	if return_pred:
		try:
			pred = Prediction.query.filter_by(mol_id = mol_id
				).filter_by(hp_settings_id = party_id).one()
		except:
			pred = None
		if not return_grade:
			return mol, pred
		else:
			return mol, pred, grade
			
	return mol


def get_ordered_molecules(screen_id, party_id, name = None, orderby = "score", mode = "annotate", modetime = None,
		limit = None, offset = None):

	all_mols = db.session.query(Molecule).all()

	"""
	Return molecules that do NOT already have grades, sorted by requested method
	"""
	if name is not None: # if molecule is requested by name, exclude nothing
		exclude = false()

	else: # either in annotation or review mode, exclude mols accordingly
		if mode == "annotate": # exclude all molecules with grades
			exclude = exists().where(Molecule.id == Grade.mol_id
				).where(Grade.hp_settings_id == party_id)
		elif mode == "review": # exclude molecules with NEW grades
			exclude = exists().where(Molecule.id == Grade.mol_id
				).where(Grade.timestamp > modetime)

	# base query - Molecules that belong to this run and don't meet the conditions above
	query = db.session.query(Molecule
		).filter(Molecule.run_id == screen_id # limit this sreen
		).filter(~exclude) 

	if name is not None: #check if we need a name filter
		# add name filter to query
		query = query.filter(Molecule.name.ilike(f"%{name}%"))

	if orderby != "score" and orderby != "name": # need to get predictions to order - uncertainty, prediction 
		query = query.outerjoin(Prediction
			).filter(Prediction.hp_settings_id == party_id)

	unordered_mols = query.offset(offset).limit(limit).all()
	
	# decide ordering - either score, uncertainty, prediction, or disagreement(error)
	if orderby != 'name':
		query = query.order_by(*orderby_dict[orderby])
	total = query.count()

	mols = query.offset(offset).limit(limit).all()

	grades = get_relations(party_id, mols, "annotations", Grade)
	preds = get_relations(party_id, mols, "uncertains", Prediction)

	return mols, grades, preds, total

def get_relations(party_id, mols, relation, dbtable):
	relations = []
	for mol in mols:
		try:
			if relation == "annotations": dbrelat = mol.annotations 
			else: dbrelat = mol.uncertains

			relat = dbrelat.filter(dbtable.hp_settings_id == party_id).one_or_none()

		except:
			relat = None
		relations.extend([relat])
	return relations

def get_predictions(party_id, sort_by = "prediction"):
	if sort_by not in ['uncertainty', 'prediction']:
		sort_by = 'prediction'

	order = [Prediction.prediction, Prediction.uncertainty]
	if sort_by != 'prediction':
		order = order[::-1]

	preds = Prediction.query.filter_by(hp_settings_id = party_id).order_by(*order).all()
	return preds

def _unpack_fp(ifp_json, fp_length = 4096):
	fp = np.zeros(fp_length)
	for key in ifp_json:
		fp[int(key)] = ifp_json[key]
	return fp

def get_num_grades(party_id):
	return Grade.query.filter_by(hp_settings_id = party_id).count()

def get_grades_for_training(party_id, format_dataframe = True, fp_col = 'fp', label_col = 'label', fp_length = 4096):
	"""
	Gets all the grades for a run, formats into expected dataframe for FingerprintDataset if requested
	"""

	grades = Grade.query.filter_by(hp_settings_id = party_id).all()

	if format_dataframe:
		data = [(int(grade.mol_id), _unpack_fp(json.loads(grade.ifp), fp_length), grade.grade) for grade in grades]
		df = pd.DataFrame(data, columns = ["id", fp_col, label_col])
		df.set_index("id", inplace=True)
		return df
	return grades

def get_molecules_for_predicting(mol_ids, format_dataframe = True, fp_col = "fp"):
	molecules = Molecule.query.filter(Molecule.id.in_(mol_ids)).all()

	if format_dataframe:
		data = [(int(mol.id), _unpack_fp(mol.ifp.counts)) for mol in molecules if mol.ifp is not None] # filter out mols with no ifp if present
		df = pd.DataFrame(data, columns = ["id", fp_col], index=mol_ids)
		return df.dropna()
	return molecules

# putting these here cause I dont know where else they fit
def prediction_csv(user_id, screen_id, party_id):
	"""
	Creates output csv file containing moelcule ids, names, grades, predictions, smiles
	"""
	prefix = "-".join([f"{val_id}{val}"for val_id, val in zip(['u', 's', 'p'], [user_id, screen_id, party_id])])

	# get existing grades - write these molecules first, and there are fewer of them so less db queries that doing it the other way around
	data = {}

	grades = get_grades_for_training(party_id, format_dataframe = False)
	for grade in grades:
		mol = grade.molecule
		data[mol.id] = [mol.name, mol.score, grade.grade, mol.smi]

	preds = get_predictions(party_id)

	for pred in preds:
		mol = pred.molecule
		if mol.id not in data.keys():
			data[mol.id] = [mol.name, mol.score, '',  mol.smi, pred.prediction, pred.uncertainty]
		else:
			data[mol.id] += [pred.prediction, pred.uncertainty]

	data_df = pd.DataFrame.from_dict(data, orient='index',
		columns = ['name', 'score', 'grade', 'smi', 'prediction', 'uncertainty'])

	data_df.to_csv(f"{UPLOAD_FOLDER}/{prefix}_predictions.csv", index_label='id')
	return True


def update_history(user_id, screen_id, hp_settings_id, update_info):

	new_model = Model(
		user_id = user_id,
		run_id = screen_id,
		hp_run_id = hp_settings_id,
		train_loss = update_info['train_loss'],
		val_loss = update_info['val_loss'],
		num_grades = update_info['train_size'])

	db.session.add(new_model)
	db.session.commit()

def recover_history(hp_settings_id):
	history_dict = {}
	history_dict["Training"] = []
	history_dict["Validation"] = []
	history_dict["Time"] = []
	try:
		# flip dict, seperate training, validation, num_sgrades
		models = Model.query.filter_by(hp_run_id = hp_settings_id
			).order_by(Model.timestamp).all()

		for model in models:
			history_dict['Training'].append(model.train_loss)
			history_dict['Validation'].append(model.val_loss)
			history_dict['Time'].append(str(model.timestamp).split('.')[0].replace(' ', '<br>')) #html syntax
		return history_dict

	except Exception as e:
		print(e)
		return {}
		