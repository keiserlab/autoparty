
import pytest

import numpy as np, torch
from sqlalchemy import inspect

from app import create_app, db
from config import config_dict as app_config_dict

from app.base.io_utils import parse_party_config
from app.base.molecule_utils import get_grades_for_training
from app.base.tasks import task_save_grade_helper
from app.base.models import FingerprintDataset, get_dataloaders, \
	BaseModel, EnsembleModel

from torch.nn import Sigmoid, Softmax

import logging
LOGGER = logging.getLogger(__name__)

@pytest.fixture
def app():
	test_config = app_config_dict['Test']
	app = create_app(test_config)
	with app.app_context():
		db.create_all() # create all tables
		yield app

@pytest.fixture
def new_model_config():
	return 'inputs/test_model.conf'

def _save_dummy_grades():
	# add 10 grades to database - 5 for first party, 5 for second
	for i, grade in enumerate(['a', 'b', 'c', 'd', 'f']):
		task_save_grade_helper(i+1, grade, 1, 1, 1, build_ifp = False, dummy_ifp = {i:i})
		task_save_grade_helper(i+1, ['a', 'c'][i % 2], 1, 1, 2, build_ifp = False, dummy_ifp = {i:i+5})

# datasets
def test_fingerprint_dataset(app):
	# assume these have expected behavior, I check them elsewhere
	_save_dummy_grades()
	first_grades = get_grades_for_training(1)
	second_grades = get_grades_for_training(2)

	# check ordinal -  df, options, fp_col = 'fp', label_col = "label", output_type = 'ordinal')
	fp_dataset = FingerprintDataset(first_grades, options = ['a', 'b', 'c', 'd', 'f'])
	assert len(fp_dataset) == 5
	assert 'converted' in fp_dataset.df.columns
	features, label = fp_dataset[0]
	assert np.array_equal(features,np.zeros(4096)) and np.array_equal(label,np.array([1, 0, 0, 0, 0]))
	features, label = fp_dataset[4]
	assert np.array_equal(label, np.array([1, 1, 1, 1, 1]))

	# check classification + different order
	fp_dataset = FingerprintDataset(first_grades, options = ['f', 'd', 'c', 'b', 'a'], output_type = "classes")
	assert np.array_equal(fp_dataset[0][1], np.array([0, 0, 0, 0, 1]))
	assert np.array_equal(fp_dataset[4][1], np.array([1, 0, 0, 0, 0]))

	# second grade, checking errors
	fp_dataset = FingerprintDataset(second_grades, options = ['a', 'b', 'c', 'd', 'f'])
	assert np.array_equal(fp_dataset[0][1], np.array([1, 0, 0, 0, 0]))
	assert np.array_equal(fp_dataset[1][1], np.array([1, 1, 1, 0, 0]))

def test_fingerprint_dataset_errors(app):
	first_grades = get_grades_for_training(1)

	with pytest.raises(Exception) as bad_column:
		fp_dataset = FingerprintDataset(first_grades, fp_col = "fingerprints")
	assert str(bad_column.value) == "Dataframe does not contain fingerprint column and/or label column."

	with pytest.raises(Exception) as bad_output_type:
		fp_dataset = FingerprintDataset(first_grades, output_type = "regression")
	assert str(bad_output_type.value) == "Invalid output type, allowed options: [ordinal, classification]"

	with pytest.raises(Exception) as no_options:
		fp_dataset = FingerprintDataset(first_grades, output_type = "ordinal")
	assert str(no_options.value) == "Must provide options for ordinal dataset."

	with pytest.raises(Exception) as illegal_label:
		fp_dataset = FingerprintDataset(first_grades, output_type = "classes", options = ["a", "b", "c"])
	assert "Not all found labels are allowed" in str(illegal_label.value)

def test_get_dataloaders_ensemble(app, new_model_config):
	# default config - ensemble classifier with 3 members, ordinal
	default_config_dict, default_errors = parse_party_config(dry_run = True)
	LOGGER.info(default_errors)
	LOGGER.info(default_config_dict)

	first_grades = get_grades_for_training(1)
	dls = get_dataloaders(first_grades, default_config_dict)
	assert len(dls) == default_config_dict['method_dict']['committee_size']
	features, labels = next(iter(dls[0]))
	assert features.shape == (5, 4096) and labels.shape == (5, 5)

	default_config_dict['method_dict']['committee_size'] = 5
	dls = get_dataloaders(first_grades, default_config_dict)
	assert len(dls) == default_config_dict['method_dict']['committee_size']

	val_dl = get_dataloaders(first_grades, default_config_dict, val = True) # return single df, no shuffle
	features1, labels1 = next(iter(val_dl))
	features2, labels2 = next(iter(val_dl))
	assert torch.equal(features1, features2) and torch.equal(labels1, labels2)

# models
def test_base_model_construction(app):
	default_config_dict, _ = parse_party_config(dry_run = True)
	base_model = BaseModel(default_config_dict)

	# architecture
	default_layers = [4096, 1024, 1024, 5]
	assert base_model.layers == default_layers
	params = list(base_model.parameters())
	assert len(params) == 6 # 3 weight matrices, 3 bias

	# other settings
	assert isinstance(base_model.net[-1], Sigmoid) # sigmoid final for ordinal outputs
	assert list(base_model.optim.param_groups)[0]['lr'] == default_config_dict['learning_rate']
	assert list(base_model.optim.param_groups)[0]['weight_decay'] == default_config_dict['weight_decay']

	# pass through
	first_grades = get_grades_for_training(1)
	test_dl = get_dataloaders(first_grades, default_config_dict, val = True) # return single df, no shuffle
	assert base_model(next(iter(test_dl))[0]).shape == (5, 5) # this also check for errors
	assert base_model.predict(next(iter(test_dl))[0]).shape == (5, 5)

	# check no uncertainty in base model
	with pytest.raises(Exception) as no_uncertainty:
		preds, uncertainties = base_model.predict(next(iter(test_dl))[0], return_uncertainty = True)
	assert str(no_uncertainty.value) == "BaseModel does not calculate uncertainty."


def test_ensemble_model_construction(app):
	default_config_dict, _ = parse_party_config(dry_run = True)
	ensemble_model = EnsembleModel(default_config_dict)

	# architecture
	default_layers = [4096, 1024, 1024, 5]
	assert len(ensemble_model.members) == default_config_dict['method_dict']['committee_size']
	assert all([member.layers == default_layers for member in ensemble_model.members])

def test_drop_all(app):
	"""
	This is run last, cleans up database.
	"""
	db.drop_all()

	assert len(inspect(db.engine).get_table_names()) == 0
	try:
		mol = Molecules.query.one()
		assert False # if we make it this far, something has done wrong
	except:
		assert True # retrieval failed, we're good
