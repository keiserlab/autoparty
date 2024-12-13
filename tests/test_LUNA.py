import os
import pytest
import logging
import time

from rdkit import Chem

from app.base.io_utils import load_remote_molecules
from app.base.luna_utils import run_mol, run_mol_batch, get_pdb_parser, parse_luna_config

LOGGER = logging.getLogger(__name__)

@pytest.fixture
def zinc_mol():
	"""
	Load static molecule and protein - in this case, using D4 data
	"""
	zinc_sdf = "inputs/ZINC000518842964.sdf"
	mol = Chem.MolFromMolFile(zinc_sdf) 
	return mol

@pytest.fixture
def zinc_mols():
	"""
	Load multiple molecules
	"""
	sdf_gz = "top20.sdf.gz"
	prot =  "inputs/3T3U.pdb"

	return sdf_gz, prot

@pytest.fixture
def pdb_path():
	return "inputs/D4.pdb"

@pytest.fixture
def malformed_pdb_path():
	return 'inputs/malformed_pdb.pdb'

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


def test_luna_default(zinc_mol, pdb_path):
	ifp, inters = run_mol("D4", "zinc-mol", zinc_mol, pdb_path)

	# check interactions - should be a single ion bridge between D115 O -- HN
	inter_types = [inter.type for inter in inters.interactions]
	assert "Ionic" in inter_types
	ion_bridge = list(inters.interactions)[inter_types.index("Ionic")]

	LOGGER.info(ion_bridge)
	lig_grp = ion_bridge.src_grp if ion_bridge.src_grp.is_hetatm() else ion_bridge.trgt_grp
	prot_grp = ion_bridge.src_grp if not ion_bridge.src_grp.is_hetatm() else ion_bridge.trgt_grp

	assert all([atom.parent.resname == 'ASP' for atom in prot_grp._atoms])
	assert all([[atom.parent.id[1] == 115 for atom in prot_grp._atoms]])
	assert all([atom.parent.resname == 'LIG' for atom in lig_grp._atoms])

	#check basic fingerprint things - length, is count fingerprint
	assert ifp.fp_length == 4096 # default length
	assert any([count > 1 for count in ifp.counts.values()])

def test_handle_bad_files():
	# we want this to throw an error, since all the actual error handling is done by the celery task
	with pytest.raises(Exception):
		ifp, inters = run_mol("", "", None, None, None)

def test_pdb_loader(pdb_path):

	assert get_pdb_parser('D4', pdb_path, dry_run = True)

	pdb_parser, structure = get_pdb_parser('D4', pdb_path, dry_run = False)
	assert pdb_parser is not None
	assert structure is not None

def _test_LUNA_config(zinc_mol, pdb_path, fp_config_path):
	# make sure various LUNA configs are actually working - messing with fp settings
	fp, inters = run_mol("D4", "zinc-mol", zinc_mol, pdb_path, {"LUNA_config":fp_config_path})
	assert fp.fp_length == 1024 # length
	assert max(fp.counts.values()) == 1 # count -> bit

def test_inter_config(hbond_inter_path):
	# run with only contains relaxed hydrogen bond constraint, should find some  new ones
	params = parse_luna_config({"inter_cfg":hbond_inter_path})
	assert len(params['inter_calc'].inter_config) == 2
	assert params['inter_calc'].inter_config['max_da_dist_hb_inter'] == 3.9
	assert params['inter_calc'].inter_config['max_ha_dist_hb_inter'] == 3


def test_mol_batch(zinc_mols):
	
	#NOTE: this test takes a few minutes to run

	mols, prot = zinc_mols
	mollist, errors = load_remote_molecules(mols, "minimizedAffinity") # tested elsewhere
	# run sequentially
	names = []
	mols = []

	seq_ifps = []

	total_seq = 0
	# run sequentially
	for mol in mollist:
		names.append(mol[0]); mols.append(Chem.MolFromMolBlock(mol[1]))
		start_time = time.time()
		ifp, inters = run_mol('prot', names[-1], mols[-1], prot)
		#print(inters.interactions)
		end_time = time.time()
		total_seq += (end_time - start_time)
		seq_ifps.append(ifp)

	# run as a batch
	total_batch = 0
	start_time = time.time()
	batch_ifps, _ = run_mol_batch('prot', names, mols, prot)	
	end_time = time.time()

	total_batch = end_time - start_time

	assert all([seq_ifps[i].counts == batch_ifps[i].counts
					for i in range(5)]) #check identical fingerprints
	assert total_batch < total_seq # check speedup
	