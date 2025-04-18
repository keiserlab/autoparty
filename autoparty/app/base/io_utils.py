#### Utility code for reading provided files, handling error checking, etc. Might migrate to celery (or add celery here)

import os
import glob
import gzip
import zipfile

import numpy as np
import pandas as pd
from rdkit import Chem

import configparser

from os.path import basename
from werkzeug.utils import secure_filename

from app.base.luna_utils import get_pdb_parser #check for malformed PDB
from app.base.defaults import DEFAULT_PARTY_CONFIG, UPLOAD_FOLDER, INPUTS_FOLDER, \
					luna_name_convert, party_config_types, party_config_options


from app.base.database import HPRunSetting


def upload_check(func):
	"""
	This handles uploading provided smaller files (mostly config/annotations) and deleting them if the check fails.
	"""
	def wrapper(*args, **kwargs):
		if 'dry_run' in kwargs and kwargs['dry_run']: # dry run for testing (testing only!), we already have the files so we don't need to upload them
			kwargs.pop('dry_run')
			if len(args) == 0:
				args = [None] # set args to be an empty list if its empty, used for default model testing
			output, errors = func(*args, **kwargs) # run check
			return output, errors

		prefix, filenames = upload_files(*args, **kwargs) # upload files
		output, errors = func(filenames, prefix = prefix) # run check, DON'T PASS KWARGS

		if errors:
			remove_files(filenames, prefix)
		return output, errors
	return wrapper

### "remote" files for use with container running on remote server
def check_remote_file(file_path):
	return os.access(f"{INPUTS_FOLDER}/{file_path}", os.R_OK)

def load_remote_pdb_file(file_path, pdb_id):
	"""
	Checks that pdb file is formed correctly and can loaded by LUNA
	Returns contents or raises exception

	TODO: At the moment this always works because LUNA has permissive=True set - do we want this?
	"""
	if not get_pdb_parser(pdb_id, f"{INPUTS_FOLDER}/{file_path}", dry_run = True, permissive = False):
		return False

	with open(f"{INPUTS_FOLDER}/{file_path}", 'r') as f:
		pdbcontents = f.read()
	return pdbcontents

def load_remote_molecules(file_path, score_label):
	
	errors = []

	root, file_extension = os.path.splitext(file_path)

	if file_extension == '.gz': # handle this case
		file_extension = os.path.splitext(root)[1] + file_extension
	if not file_extension in ['.sdf', '.sdf.gz']:
		errors.append("Illegal file extension for molecules file.")
		return [], errors

	mollist = []
	fail_load = 0

	file_path = f"{INPUTS_FOLDER}/{file_path}"

	if 'sdf' in file_extension:
		if 'gz' in file_extension:
			with gzip.open(file_path) as gzf:
				supl = Chem.ForwardSDMolSupplier(gzf)
				mols = [x for x in supl if x is not None]
		else:
			supl = Chem.ForwardSDMolSupplier(file_path) 
			mols = [x for x in supl if x is not None]

	for mol in mols:
		try:
			mollist.append((mol.GetProp("_Name"), Chem.MolToMolBlock(mol), mol.GetProp(score_label)))
		except:
			fail_load += 1

	if fail_load:
		errors.append(f"{fail_load} molecules failed to load - double check provided score label and molecules file.")
	return mollist, errors

@upload_check
def load_existing_annotations(filenames, prefix = "", name_col = "name", grade_col = "grade"):
	"""
	Takes in path to file, processes annotations and returns grades, annotations
	"""
	if isinstance(filenames, list):
		filename = filenames[0]
	else:
		filename = filenames

	csv_name = f"{prefix}{filename}"
	grades = pd.read_csv(csv_name, 
		index_col = name_col, 
		low_memory = True)
	meta_cols = [col for col in grades.columns if col.startswith("meta_")]
	if not grade_col in grades.columns:
		keep_cols = meta_cols
	else:
		keep_cols = [grade_col] + meta_cols

	grades = grades[keep_cols]
	grades.index = grades.index.astype(str, copy = False)
	grades.dropna(inplace = True, how = 'all') # drop anything with no information

	if len(meta_cols) == 0: meta_cols = None # makes downstream things easier
	return grades, meta_cols


# Model config parsing
def _string_to_list(input_string):

	option_list = []
	split_str = input_string.split(',')
	for elem in split_str:
		elem = elem.strip()
		if len(elem) == 0: raise Exception("At least one empty element in list.")
		#try:
		#	option_list.append(int(elem))
		#except:
		option_list.append(elem) # for now, using strings only
	if not len(set(option_list)) == len(option_list):
		raise Exception("Duplicate outputs in list.")

	return option_list

def _check_type(x, option_name):
	try:
		if option_name == 'output_options':
			x = _string_to_list(x)
		return party_config_types[option_name](x)
	except:
		return None

def _check_options(x, option_name):
	if not x in party_config_options[option_name]:
		return f"Error in {option_name}, {x} not in {party_config_options[option_name]}"
	else: return None

@upload_check
def parse_party_config(filenames, prefix = "", default_config_file_path = DEFAULT_PARTY_CONFIG):
	# handle that filename may be None, a list with None, a string, or a list with one string, depending on how this function is called
	filename = None
	if filenames:
		if isinstance(filenames, list):
			filename = filenames[0]
		else:
			filename = filenames

	config_errors = {}
	config_dict = {}

	config = configparser.ConfigParser()
	config.read(default_config_file_path) # load default model config

	# override with updated config file, if provided
	if filename:
		config.read(f"{prefix}{filename}")


	# validate party config
	for key, val in config.items('model'):
		new_val = _check_type(val, key)
		if new_val is not None: 
			# need to validate more if there are choices
			if key in party_config_options:
				option_check = _check_options(new_val, key)
				if option_check: config_errors[key] = f"[model] {key} " + option_check ; continue

			config_dict[key] = new_val
		else:
			config_errors[key] = f"[model] {key}: Error converting {val} to {party_config_types[key]}"

	#TODO: I only check correctness for applicable parameters - is this ok?
	method_dict = {}
	if 'uncertainty_method' in config_dict:
		# validate uncertainty-specific config
		for key, val in config.items(config_dict['uncertainty_method']): # should be able to find it, validated options by this point
			new_val = _check_type(val, key)
			if new_val is not None: method_dict[key] = new_val
			else: config_errors[key] = f"[{config_dict['uncertainty_method']}]{key}: Error converting {val} to {party_config_types[key]}"
	
	config_dict['method_dict'] = method_dict
	return config_dict, config_errors


def _get_file_match(file_match, file_list):
	"""
	Finds single file matching patters in file list, else throws an error.
	"""
	file_index = [i for i, file_name in enumerate(file_list) if file_name.endswith(file_match)]
	if len(file_index) > 1:
		raise ValueError(f"More than one {luna_name_convert[file_match]} found.")
	return None if len(file_index) == 0 else file_list[file_index[0]] # None if not found, sets to default later

@upload_check
def check_luna_config(file_list, prefix = ""):
	# dry_run used for testing only! shouldn't be used without upload check for actual execution
	### general LUNA configs
	luna_errors = []
	accepted_files = []
	luna_files = {}

	### set all other potential uploaded files
	for file_name, file_key in luna_name_convert.items():
		try:
			match = _get_file_match(file_name, file_list)
			if match is not None:
				accepted_files.append(match)
				luna_files[file_key] = f"{prefix}{match}"
		except ValueError as e: # catch errors, append to error list
			luna_errors.append(str(e))
	if len(accepted_files) != len(file_list):
		luna_errors.append("Some provided files not being used: " + ",".join(list(set(file_list) - set(accepted_files)))) # looks like lisp
	return luna_files, luna_errors

# saves provided file from the user into uploads folder, renames
def upload_files(request_files, user_id = -1, screen_id = -1, party_id = -1, destination_dir = UPLOAD_FOLDER, new_filename = ""):
	filenames = []
	prefix =  "-".join([f"{val_id}{val}"for val_id, val in zip(['u', 's', 'p'], [user_id, screen_id, party_id]) if val >= 0]) + "_"
	for file in request_files:
		if len(file.filename) == 0:
			continue
		if not len(new_filename):
			base = basename(file.filename)
		else:
			base = new_filename
		target = os.path.join(destination_dir, f"{prefix}{base}")
		file.save(target)
		filenames.append(base)
	return f"{destination_dir}/{prefix}", filenames

# removes files if they fail correctness check
def remove_files(filenames, prefix, suffix = "", inverse = False):
	if inverse: # delete everything in file that doesnt have filename in it
		for file in glob.glob(prefix + "*" + suffix):
			if not any([filename in file for filename in filenames]):
				os.remove(file)
	else:
		for file in filenames:
			os.remove(f"{prefix}{file}")

def recover_curves(user_id, screen_id, party_id):

	loaded_curves = []
	prefix =  "-".join([f"{val_id}{val}"for val_id, val in zip(['u', 's', 'p'], [user_id, screen_id, party_id])]) + "_"
	party_files = glob.glob(f"{UPLOAD_FOLDER}/{prefix}*.npy")

	if len(party_files) > 2:
		return None, None # something went wrong, don't recover history
		#TODO: handle this case better
	for file in sorted(party_files): # training, validation
		loaded_curves.append(np.load(file))

	while len(loaded_curves) < 2:
		loaded_curves.append(None)

	return loaded_curves

def prepare_files(user_id, screen_id, party_id):
	"""
	File utils to create zip file with LUNA config, party_config, models, loss_curves 
	Done async, triggered on model training. Might move to IO utils
	"""
	prefix = "-".join([f"{val_id}{val}"for val_id, val in zip(['u', 's', 'p'], [user_id, screen_id, party_id])])
	
	files = []
	for file in os.listdir(UPLOAD_FOLDER):
		file_prefix = file.split('_')[0]
		if len(file_prefix) > 0 and (file_prefix in prefix) and ('results.zip' not in file):
			files.append(f"{UPLOAD_FOLDER}/{file}") 

	if not any([x.endswith('party.conf') for x in files]): # add default party config
		files.append(DEFAULT_PARTY_CONFIG)

	newzip = zipfile.ZipFile(f"{UPLOAD_FOLDER}/{prefix}_results.zip", "w" )

	# there are two potential files - the first is 
	for a in files:
		filename = "_".join([token for token in os.path.basename(a).split('_')[1:]])
		newzip.write(a, filename, compress_type=zipfile.ZIP_DEFLATED)
	newzip.close()
	return True