import pytest

import filecmp

from flask import Request
from werkzeug.datastructures import FileStorage

from app.base.io_utils import * 
from app.base.io_utils import _string_to_list # import private function, we need it
from app.base.defaults import *

import logging
LOGGER = logging.getLogger(__name__)

## fixtures - hard coded test file paths
@pytest.fixture
def example_sdf():
	return 'testset_docked_best.sdf'

@pytest.fixture
def example_sdf_gz():
	return 'top20.sdf.gz'

@pytest.fixture
def example_pdb():
	return "D4.pdb"

@pytest.fixture
def dummy_file():
	return 'dummy_file.txt'

@pytest.fixture
def example_annotations():
	return [f'inputs/{annot}.csv' for annot in ['grade_annot', 'meta_annot', 'annot']]

@pytest.fixture
def new_model_config():
	return 'inputs/test_model.conf'

@pytest.fixture
def bad_model_config():
	return 'inputs/bad_model.conf'

@pytest.fixture
def fp_config_path():
	return 'inputs/luna_test_configs/ifp_luna.cfg'

@pytest.fixture
def hbond_inter_path():
	return 'inputs/luna_test_configs/hbond_inter.cfg'

@pytest.fixture
def ppi_filter_path():
	return 'inputs/luna_test_configs/ppi_filter.cfg'

@pytest.fixture
def ionic_config_path():
	return "inputs/luna_test_configs/ionic_bind.cfg"


## tests
### "remote' files
def test_check_remote_file(example_sdf, dummy_file):
	assert check_remote_file(example_sdf) #can we read real files
	assert not check_remote_file(dummy_file) # we shouldnt be able to read garbage

def test_load_remote_sdf(example_sdf):
	# good run
	mols, failure = load_remote_molecules(example_sdf, "minimizedAffinity")
	assert len(mols) == 22
	assert len(failure) == 0

def test_load_remote_sdfgz(example_sdf_gz):
	mols, failure = load_remote_molecules(example_sdf_gz, "minimizedAffinity")
	assert len(mols) == 20
	assert len(failure) == 0

def test_wrong_score_name(example_sdf):
	"""
	TODO: think about error checking here - do we want to load the files anyway? What if only certain mols have real scores?
	"""
	# bad run
	mols, failure = load_remote_molecules(example_sdf, "miinmizedAffinity")
	assert len(mols) == 0
	assert '22' in failure[0]

def test_illegal_remote_file(dummy_file):
	mols, failure = load_remote_molecules(dummy_file, "")
	assert len(mols) == 0
	assert len(failure) == 1
	assert failure[0] == "Illegal file extension for molecules file."

def test_load_remote_pdb(example_pdb):
	pdbcontents = load_remote_pdb_file(example_pdb, 'D4')
	assert len(pdbcontents.split("\n")) == 6145
	assert pdbcontents.split("\n")[0] == "HEADER    SIGNALING PROTEIN/ANTAGONIST            20-JUL-17   5WIU              "
	assert pdbcontents.split("\n")[-2] == "END"     

def _test_load_malformed_pdb():
	# currently under operation
	raise NotImplementedError        

### Uploading existing grades/metadata
def test_upload_annotations_grades_only(example_annotations):
	grade_only, _, _ = example_annotations
	
	grades, metacols = load_existing_annotations(grade_only, dry_run = True)
	assert metacols is None
	assert "grade" in grades.columns
	assert all([grades.at[f"Molecule{idx}", "grade"] == exp for idx, exp in zip(list(range(1,6)),['A', 'B', 'C', 'D', 'E'])])

def test_upload_annotations_meta_only(example_annotations):
	_, meta_only, _ = example_annotations
	
	grades, metacols = load_existing_annotations(meta_only, dry_run = True)
	assert metacols is not None
	assert "grade" not in grades.columns
	assert all([grades.at[f"Molecule{idx}", metacols[0]] == exp for idx, exp in zip(list(range(1,6)),list(range(10, 51, 10)))])

def test_upload_annotations_all(example_annotations):
	_, _, all_annot = example_annotations
	
	grades, metacols = load_existing_annotations(all_annot, dry_run = True)
	assert metacols is not None
	assert "grade" in grades.columns
	assert all([grades.at[f"Molecule{idx}", metacols[0]] == exp for idx, exp in zip(list(range(1,6)),list(range(10, 51, 10)))] + \
		[grades.at[f"Molecule{idx}", metacols[0]] == exp for idx, exp in zip(list(range(1,6)),list(range(10, 51, 10)))])

### Model config 
def test_default_model_config():
	config_dict, errors = parse_party_config(dry_run = True)
	assert not errors # should be no errors from default dict
	assert config_dict['uncertainty_method'] == 'ensemble'
	assert 'committee_size' in config_dict['method_dict'] and 'data_split' in config_dict['method_dict']

	# TODO: check all the values?

def test_provided_model_config(new_model_config):
	config_dict, errors = parse_party_config(new_model_config, dry_run = True)
	assert not errors # should be no errors from default dict
	assert config_dict['uncertainty_method'] == 'dropout'
	assert 'passes' in config_dict['method_dict']

	assert config_dict['learning_rate'] == 1e-6
	assert config_dict['hidden_layers'] == 3
	assert config_dict['weight_decay'] == 1e-2
	assert config_dict['output_options'] == ['1', '2', '3'] # TODO: is it okay to keep these as strings? or should be try to conver to ints?
	assert config_dict['output_type'] == 'classes'

def test_check_options():
	assert _string_to_list("a,b,c,d,f") == ["a", "b", "c", "d", "f"]
	assert _string_to_list("1,2,3,4,5") == ['1', '2', '3', '4', '5']
	assert _string_to_list("1, 2, 3, 4, 5") == ['1', '2', '3', '4', '5']

	with pytest.raises(Exception) as check_empty:
		_string_to_list("1,2,,4,5")
	assert str(check_empty.value) == "At least one empty element in list."

	with pytest.raises(Exception) as check_duplicate:
		_string_to_list("1,2,2,4,5")
	assert str(check_duplicate.value) == "Duplicate outputs in list."

def test_malformed_model_config(bad_model_config):
	config_dict, errors = parse_party_config(bad_model_config, dry_run = True)

	assert errors
	assert all([name in errors for name in ['learning_rate', 'n_neurons', 'output_type', 'output_options']])
	assert 'dropout' not in errors and 'dropout' in config_dict

def test_check_luna_config():
	# needs to check with correct file and incorrect files
	
	# correct files
	test_config_list = ["test_luna.cfg", "test_filter.cfg", "test_inter.cfg", "test_bind.cfg"]
	test_config_dict, errors = check_luna_config(test_config_list, dry_run = True)

	assert not errors
	assert len(test_config_list) == len(test_config_dict)
	assert all([test_config_dict[luna_name_convert[test_config_list[i].split("_")[1]]] == test_config_list[i] for i in range(len(test_config_dict))]) # messy

	# check prefix
	prefix = "prefix_"
	test_config_list = ["test_luna.cfg", "test_filter.cfg", "test_inter.cfg", "test_bind.cfg"]
	test_config_dict, errors = check_luna_config(test_config_list, prefix = prefix, dry_run = True)

	print(test_config_dict)

	assert not errors
	assert len(test_config_list) == len(test_config_dict)
	assert all([test_config_dict[luna_name_convert[test_config_list[i].split("_")[1]]] == f"{prefix}{test_config_list[i]}" for i in range(len(test_config_dict))]) # messier

	# incorrect files
	# duplicate values
	test_config_list = ["test_1_inter.cfg", "test_2_inter.cfg"]
	test_config_dict, errors = check_luna_config(test_config_list, dry_run = True)

	assert len(errors) == 2 # duplicate files, files not being used
	assert errors[0] == "More than one inter_cfg found."


	# incorrectly named files
	test_config_list = ["test_luna.cfg", "test_filter.cfg", "test_inter.cfg", "test_binder.cfg", "atom_def.fdaf"]
	test_config_dict, errors = check_luna_config(test_config_list, dry_run = True)

	assert len(errors) == 1
	assert errors[0].startswith("Some provided files not being used:") 
	assert "test_binder.cfg" in errors[0]
	assert "atom_def.fdaf" in errors[0]
	assert "test_inter.cfg" in test_config_dict.values() and "test_inter.cfg" not in errors[0]

def test_upload_files(new_model_config):
	# move file into uploads folder, check that it's there, check that filename is structured correctly

	# upload expects a FileStorage object under the hood, so we have to recreate that here
	with open(new_model_config, 'rb') as f:
		file = FileStorage(f)
		prefix, filenames = upload_files([file], user_id = 1, screen_id = 1) # needs to be run with file open, handled automatically when in request

	assert prefix == f"{UPLOAD_FOLDER}/u1-s1_"
	assert len(filenames) == 1
	assert filenames[0] == new_model_config.split('/')[-1]

	assert os.path.exists(f"{prefix}{filenames[0]}")
	assert filecmp.cmp(new_model_config, f"{prefix}{filenames[0]}")

def test_upload_check(fp_config_path, hbond_inter_path, ppi_filter_path, ionic_config_path, 
		new_model_config, bad_model_config):
	# check that entire decorator works properly
	# real files, working with luna_config_files
	test_config_list = [fp_config_path, hbond_inter_path, ppi_filter_path, ionic_config_path]

	# same as above, need to create FileStorage objects
	files = []
	with open(fp_config_path, 'rb') as fp, \
		 open(hbond_inter_path, 'rb') as hbond, \
		 open(ppi_filter_path, 'rb') as ppi, \
		 open(ionic_config_path, 'rb') as ion:

		files.append(FileStorage(fp))
		files.append(FileStorage(hbond))
		files.append(FileStorage(ppi))
		files.append(FileStorage(ion))

		test_config_dict, errors = check_luna_config(files, user_id = 1, screen_id = 1)

	# check luna_config is correct
	prefix = f"{UPLOAD_FOLDER}/u1-s1_"

	assert not errors
	assert len(test_config_list) == len(test_config_dict)
	assert all([test_config_dict[luna_name_convert[test_config_list[i].split("_")[-1]]] == 
		f"{prefix}{os.path.basename(test_config_list[i])}" for i in range(len(test_config_dict))]) # messy

	# check that all files have been copied over successfully
	for val, path in zip(sorted(test_config_dict.values()), sorted(test_config_list)):	
		assert os.path.exists(val)
		assert filecmp.cmp(val, path)

	# also checking with model config
	with open(new_model_config, 'rb') as f:
		file = FileStorage(f)
		config_dict, errors = parse_party_config([file], user_id = 1, screen_id = 1)
	
	# check config dict
	assert not errors
	assert config_dict['uncertainty_method'] == 'dropout'
	assert 'passes' in config_dict['method_dict']

	assert config_dict['learning_rate'] == 1e-6
	assert config_dict['hidden_layers'] == 3
	assert config_dict['weight_decay'] == 1e-2
	assert config_dict['output_options'] == ['1', '2', '3'] # TODO: is it okay to keep these as strings? or should be try to conver to ints?
	assert config_dict['output_type'] == 'classes'

	# check file comparison
	assert filecmp.cmp(new_model_config, f"{prefix}{os.path.basename(new_model_config)}")

	# incorrect / malformed files
	with open(bad_model_config, 'rb') as f:
		file = FileStorage(f)
		config_dict, errors = parse_party_config([file], user_id = 1, screen_id = 1)
	assert all([name in errors for name in ['learning_rate', 'n_neurons', 'output_type', 'output_options']])
	assert 'dropout' not in errors and 'dropout' in config_dict

	# check that file doesn't exist
	assert not os.path.exists(f"{prefix}{os.path.basename(bad_model_config)}")

def test_remove_files(fp_config_path, hbond_inter_path, ppi_filter_path, ionic_config_path, 
		new_model_config):
	# remove files from test_upload_files, clean up
	filenames = [os.path.basename(f) for f in [fp_config_path, hbond_inter_path, ppi_filter_path, ionic_config_path, 
		new_model_config]]

	prefix = f"{UPLOAD_FOLDER}/u1-s1_"
	remove_files(filenames, prefix)
	assert all([not os.path.exists(f"{prefix}{f}") for f in filenames])