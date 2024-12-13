
import pytest

import json
import logging

from rdkit import Chem

from app import create_app, db
from config import config_dict

from app.base.database import Molecule, Grade, Prediction
from app.base.tasks import save_molecule, save_prediction, task_save_grade_helper
from app.base.molecule_utils import get_molecule_by_id, get_grades_for_training, get_molecules_for_predicting
from app.base.molecule_utils import get_ordered_molecules

from datetime import datetime

LOGGER = logging.getLogger(__name__)

@pytest.fixture
def app():
	test_config = config_dict['Test']
	app = create_app(test_config)
	with app.app_context():
		db.create_all() # create all tables
		yield app


def test_save_molecule(app):
	"""
	Create 20 dummy molecules and save to database, 2 screens
	"""
	for i in range(20):
		save_molecule(f'mol{i}', # mol_name
			int(i > 9) + 1,  # screen_id
			Chem.MolFromSmiles('Cc1ccccc1'), 
			i, #score 
			{}, #interaction json
			None) # ifp

	# assert 20 mols in database
	mols = Molecule.query.all()

	assert len(mols) == 20
	assert set([int(mol.run_id) for mol in mols]) == {1, 2}

def test_get_mol_by_id(app):
	mol = get_molecule_by_id(1)
	assert mol.id == 1
	assert mol.run_id == 1

def test_save_grade_helper_grades(app):
	#should be no grades at the start
	assert len(Grade.query.all()) == 0

	#insert 5 grades for first screen
	#mol_id, grade, run_id, user_id, party_id
	for i, grade in enumerate(['a', 'b', 'c', 'd', 'e']):
		task_save_grade_helper(i+1, grade, 1, 1, 1, build_ifp = False, dummy_ifp = {i:i})

	grades = Grade.query.all()
	assert len(grades) == 5
	assert set([grade.grade for grade in grades]) == {'a', 'b', 'c', 'd', 'e'}

	# check overriding and adding illegal grade
	task_save_grade_helper(5, None, 1, 1, 1, build_ifp = False)
	task_save_grade_helper(6, None, 1, 1, 1,  build_ifp = False)

	grades = Grade.query.all()
	assert len(grades) == 5
	assert set([grade.grade for grade in grades]) == {'a', 'b', 'c', 'd', 'e'}

	# check overwrite existing grades
	for i in range(1,5):
		task_save_grade_helper(i+1, 'c', 1, 1, 1, build_ifp = False)

	grades = Grade.query.all()
	assert len(grades) == 5
	assert set([grade.grade for grade in grades]) == {'a', 'c'}

	# At the end of this function, grades for screen 1 are [a, c, c, c, a]

def test_get_mol_and_grade_by_id(app):
	# working case
	mol, grade = get_molecule_by_id(1, return_grade = True, party_id = 1)
	assert mol.id == 1
	assert grade.grade == 'a'

	# not assigned a grade yet
	mol, grade = get_molecule_by_id(6, return_grade = True, party_id = 1)
	assert mol.id == 6
	assert grade is None

	# wrong party
	mol, grade = get_molecule_by_id(1, return_grade = True, party_id = 2)
	assert mol.id == 1
	assert grade is None

	# forgot party argument
	mol, grade = get_molecule_by_id(1, return_grade = True)
	assert mol.id == 1
	assert grade is None

def test_get_molecules_by_score(app): #get_molecules_by_score(run_id, party_id, exclude_annotated = True, limit = None
	# get molecules as normal 
	mols, _, _, _ = get_ordered_molecules(1, 1)
	assert len(mols) == 5 #should be unnannotated molecules only
	assert set([mol.run_id for mol in mols]) == {1} # right screen
	assert all([mols[i].score < mols[i+1].score for i in range(len(mols) - 1)]) # check order

	assert len(get_ordered_molecules(1, 2)[0]) == 10 # should get all the molecules, none have grades in hp2
	assert len(get_ordered_molecules(2, 1)[0]) == 10 # should get all the molecules,

	all_mols = get_ordered_molecules(1, 2)[0]
	offset_mols = get_ordered_molecules(1, 2, offset = 5, limit = 2)[0]

	assert len(offset_mols) == 2
	assert all_mols[5] == offset_mols[0] # check that offset was set correctly

	assert len(get_ordered_molecules(1, 1, mode = 'review', modetime = datetime.now())[0]) == 10
	assert len(get_ordered_molecules(1, 1, mode = 'review', limit = 2, modetime = datetime.now())[0]) == 2

def test_save_predictions(app):
	#mol_id, user_id, party_id, prediction, var
	assert len(Prediction.query.all()) == 0

	# assign predictions to 10 molecules (5 with grades, 5 without)
	for i in range(1, 11):
		save_prediction(i, 1, 1, ['c', 'a'][i % 2], float(i/10), 0) # -> ['a', 'c', 'a', 'c'...]

	preds = Prediction.query.all()
	assert len(preds) == 10
	assert set([pred.prediction for pred in preds]) == {'a', 'c'}
	assert set([pred.user_id for pred in preds]) == {1}
	assert set([pred.hp_settings_id for pred in preds]) == {1}

def test_get_molecules_by_uncertainty(app):
	#party_id, exclude_annotated = True, limit = None, return_preds = False
	mols, _, preds, _ = get_ordered_molecules(1, 1, orderby = 'uncertainty')
	assert len(preds) == 5
	assert set([pred.hp_settings_id for pred in preds]) == {1}
	assert all([preds[i].uncertainty > preds[i+1].uncertainty \
		for i in range(len(preds) - 1)]) # check order

	assert len(get_ordered_molecules(1, 1, mode = 'review', modetime = datetime.now())[0]) == 10 # should have gotten all of them

def test_get_molecules_by_prediction(app):
	#party_id, exclude_annotated = True, limit = None, return_preds = False
	mols, _, preds, _ = get_ordered_molecules(1, 1, orderby = 'prediction')
	assert len(preds) == 5 # 5 molecules have predictions but no grades
	assert set([pred.hp_settings_id for pred in preds]) == {1} # all from same run
	assert all([pred.prediction == p for pred, p in zip(preds, ['a', 'a', 'c', 'c', 'c'])])

	mols_only = get_ordered_molecules(1, 1, orderby = 'prediction')[0]
	assert all([mols_only[i].id == mols[i].id for i in range(len(mols_only))]) # make sure order is still by prediction


def test_get_predictions(app):

	preds = get_ordered_molecules(1, 1, orderby = 'prediction', mode='review', modetime=datetime.now())[2]
	assert all([preds[i].prediction == 'a' for i in range(5)]) and all([preds[i].prediction == 'c' for i in range(5, 10)])
	assert preds[0].uncertainty == 0.1 # we want the most sure first
	assert preds[5].uncertainty == 0.2 # should be sorted in ascending uncertainty within group

	preds = get_ordered_molecules(1, 1, orderby = 'uncertainty')[2]
	assert all([preds[i].uncertainty > preds[i+1].uncertainty for i in range(len(preds) - 1)]) # check order
	assert preds[0].prediction == 'c' and preds[1].prediction == 'a' # we want the most sure first


def test_get_grades(app):
	grades = get_grades_for_training(1, format_dataframe = False)
	assert len(grades) == 5
	assert set([grade.grade for grade in grades]) == {'a', 'c'}
	assert set([grade.hp_settings_id for grade in grades]) == {1}

	grades = get_grades_for_training(2, format_dataframe = False)
	assert len(grades) == 0

	grade_df = get_grades_for_training(1)
	assert grade_df.shape == (5, 2)
	assert all([grade_df.at[i, 'fp'][i-1] == i-1 for i in range(1,6)])
	assert grade_df.at[1, 'label'] == 'a' and grade_df.at[2, 'label'] == 'c'

def test_get_molecules_for_predicting(app):
	mol_df = get_molecules_for_predicting([1, 2, 3, 4])
	print(mol_df.columns, mol_df.shape)
	assert mol_df.shape == (0,2)

	molecules = get_molecules_for_predicting([1,2,3,4], format_dataframe = False)
	assert len(molecules) == 4

def test_drop_all(app):
	"""
	This is run last, cleans up database.
	"""
	db.drop_all()

	assert len(db.engine.table_names()) == 0
	try:
		mol = Molecules.query.one()
		assert False # if we make it this far, something has done wrong
	except:
		assert True # retrieval failed, we're good