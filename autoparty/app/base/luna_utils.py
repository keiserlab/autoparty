
import os
import sys

from app.base.defaults import LUNA_default_configs, luna_name_convert

import configparser
import networkx as nx

import luna

from rdkit.Chem import ChemicalFeatures

from luna.projects import StructureCache 
from luna.mol.entry import MolFileEntry
from luna.mol.features import FeatureExtractor
from luna.mol.groups import AtomGroupPerceiver
from luna.config.params import ProjectParams
from luna.interaction.contact import get_contacts_with
from luna.interaction.calc import InteractionCalculator, InteractionsManager
from luna.interaction.filter import InteractionFilter, BindingModeFilter
from luna.interaction.fp.shell import ShellGenerator
from luna.interaction.fp.type import IFPType
from luna.util.default_values import *
from luna.MyBio.util import get_entity_from_entry
from luna.MyBio.PDB.PDBParser import PDBParser

from luna.util import rgb2hex

import openbabel.pybel
openbabel.pybel.ob.obErrorLog.StopLogging()

IFP_TYPES = {
    "EIFP": IFPType.EIFP,
    "FIFP": IFPType.FIFP,
    "HIFP": IFPType.HIFP,
}

def run_mol_batch(prot_id, zids, loaded_mols, protein, 
    luna_config_paths = {}, celery_task = None):

    params = parse_luna_config(luna_config_paths)

    # running multiple molecules with the same protein, cache protein properties
    first_entry = MolFileEntry.from_mol_obj(prot_id, zids[0],
                    loaded_mols[0])
    first_entry.pdb_file = protein

    cache = cache_protein_properties(first_entry, prot_id, protein, params)

    inters = []
    ifps = []

    for zid, loaded_mol in zip(zids, loaded_mols):
        ifp, inter = run_mol(prot_id, zid, loaded_mol, protein, 
                                params = params, 
                                cache = cache)
        inters.append(inter)
        ifps.append(ifp)

        if celery_task:
            celery_task.update_state(state = "PROGRESS", meta = {"complete": len(ifps), "total": len(zids)})
    return ifps, inters

def get_protein_cache(prot_id, zid, loaded_mol, protein, 
    luna_config_paths={}):
    params = parse_luna_config(luna_config_paths)

    # running multiple molecules with the same protein, cache protein properties
    first_entry = MolFileEntry.from_mol_obj(prot_id, zid,
                    loaded_mol)
    first_entry.pdb_file = protein

    cache = cache_protein_properties(first_entry, prot_id, protein, params)
    return cache, params

#runname, mol_name, mol, pdb_file, luna_config
def run_mol(prot_id, zid, loaded_mol, protein, 
                    luna_config_paths = {}, 
                    params = None,
                    cache = None):
    """
    Runs LUNA to calculate the interactions and interaction fingerprint for a single protein-ligand pair.
    """
    if not params:
        params = parse_luna_config(luna_config_paths)

    entry = MolFileEntry.from_mol_obj(prot_id, zid,
                    loaded_mol)
    # set up protein
    entry.pdb_file = protein
    
    pdb_parser, structure = get_pdb_parser(entry.pdb_id, protein, dry_run = False)
    add_hydrogen = decide_hydrogen_addition(True, pdb_parser.get_header(), entry)

    structure = entry.get_biopython_structure(structure, pdb_parser)
    
    ligand = get_entity_from_entry(structure, entry)
    ligand.set_as_target(is_target = True)

    atm_grps_mngr = perceive_chemical_groups(entry, structure[0], 
                                                ligand, params, 
                                                add_hydrogen, cache)
    atm_grps_mngr.entry = entry
    
    calc_func = params['inter_calc'].calc_interactions
    interactions_mngr = calc_func(atm_grps_mngr.atm_grps)
    interactions_mngr.entry = entry

    atm_grps_mngr.merge_hydrophobic_atoms(interactions_mngr)

    # updated parsing - work with uploaded LUNA config files
    bm_filter_func = interactions_mngr.filter_out_by_binding_mode
    bm_filter_func(params["binding_mode_filter"])
    
    ifp = create_ifp(atm_grps_mngr, params)

    return ifp, interactions_mngr

def get_pdb_parser(pdb_id, pdb_file, dry_run = True, permissive = True):
    pdb_parser = PDBParser(PERMISSIVE=permissive, QUIET=True, 
                                        FIX_EMPTY_CHAINS=True,
                                        FIX_ATOM_NAME_CONFLICT=True, 
                                        FIX_OBABEL_FLAGS=False)

    structure = pdb_parser.get_structure(pdb_id, pdb_file) #check

    if not dry_run: return pdb_parser, structure  
    else: return True  
    
# parse any provided LUNA config files, using existing LUNA config checking rather than writing my own
def parse_luna_config(luna_config_paths):
    # takes in list of LUNA files (already located in UPLOAD_FOLDER, and all paths absolute to that), which can contain:
    # 1. general config (*_luna.conf)
    # 2. atom_prop_file (*.fdef) 
    # 3. interaction_config (*inter.conf) - interaction angles, distances
    # 4. filter_config (*filter.conf) - what kinds of interactions: protein-ligand, protein-protein, 
    # 5. binding mode config (*_bind.conf)
    # and creates corresponding LUNA ProjectParams to use during interaction calculation

    luna_config_dict = {}

    # Overall LUNA configs - load default, followed by provided (if it exists)
    luna_config_parser = configparser.ConfigParser()
    luna_config_parser.read(LUNA_default_configs['LUNA_config'])
    if 'LUNA_config' in luna_config_paths:
        luna_config_parser.read(luna_config_paths['LUNA_config'])

    for section in luna_config_parser.sections():
        luna_config_dict.update(luna_config_parser.items(section))
    
    # set other filepaths, either provided + copied to inputs or read from default
    for luna_file, file_path in LUNA_default_configs.items():
        if luna_file == "LUNA_config":
            continue
        if luna_file in luna_config_paths:
            luna_config_dict[luna_file] = luna_config_paths[luna_file]
        else:
            luna_config_dict[luna_file] = LUNA_default_configs[luna_file]

    # now update the remainder, fill defaults for things not provided
    params = ProjectParams(luna_config_dict, fill_defaults = True)
    return params

# calculation utils
def decide_hydrogen_addition(try_h_addition, pdb_header, entry):
    if try_h_addition:
        if "structure_method" in pdb_header:
            method = pdb_header["structure_method"]
            # If the method is not a NMR type does not add hydrogen as it usually already has hydrogens.
            if method.upper() in NMR_METHODS:
                return False
        return True
    return False

def get_perceiver(params, add_h = False, cache = None):
    feats_factory_func = ChemicalFeatures.BuildFeatureFactory
    feature_factory = feats_factory_func(params['atom_prop_file'])
    feature_extractor = FeatureExtractor(feature_factory)

    perceiver = AtomGroupPerceiver(feature_extractor, 
        add_h=add_h, ph=params['ph'], 
        amend_mol=params['amend_mol'], 
        #mol_obj_type=params['mol_obj_type'], 
        cache = cache, 
        tmp_path='/tmp')
    return perceiver

def cache_protein_properties(entry, pdb_id, protein, params):
    
    pdb_parser, structure = get_pdb_parser(entry.pdb_id, protein, dry_run = False)

    structure = entry.get_biopython_structure(structure, pdb_parser)
    add_hydrogen = decide_hydrogen_addition(True, pdb_parser.get_header(), entry)
    
    ligand = get_entity_from_entry(structure, entry)
    ligand.set_as_target(is_target = True)

    radius = params['inter_calc'].inter_config.get("cache_cutoff",
                                  BOUNDARY_CONFIG["cache_cutoff"])
    
    nb_pairs = get_contacts_with(structure[0], ligand,
                                     level='R', radius=radius)
    nb_compounds = set([p[0] for p in nb_pairs
                            if not p[0].is_target()])

    mol_objs_dict = {}
    if isinstance(entry, MolFileEntry):
            mol_objs_dict[entry.get_biopython_key()] = entry.mol_obj

    perceiver = get_perceiver(params, add_h = add_hydrogen)
    atm_grps_mngr = perceiver.perceive_atom_groups(nb_compounds,
                                                    mol_objs_dict=mol_objs_dict)

    valid_edges = set()
    for edge in atm_grps_mngr.graph.edges:
        if any([atm.parent.is_target() for atm in edge]) is False:
            valid_edges.add(edge)
    atm_grps_mngr.graph = nx.Graph()
    atm_grps_mngr.graph.add_edges_from(valid_edges)

    # Remove dummy chain if the ligand comes from an
    # external molecular file.
    if isinstance(entry, MolFileEntry):
        chain = ligand.parent
        chain.detach_child(ligand.id)

        if len(chain.child_list) == 0:
            chain.parent.detach_child(chain.id)

    return StructureCache(nb_compounds, atm_grps_mngr)

def perceive_chemical_groups(entry, entity, ligand, params, add_h=False, cache = None):

    perceiver = get_perceiver(params, add_h = add_h, cache = cache)

    radius = params['inter_calc'].inter_config.get("bsite_cutoff", BOUNDARY_CONFIG["bsite_cutoff"])
    nb_pairs = get_contacts_with(entity, ligand, level='R', radius=radius)
    nb_compounds = set([x[0] for x in nb_pairs])

    mol_objs_dict = {}
    if isinstance(entry, MolFileEntry):
        mol_objs_dict[entry.get_biopython_key()] = entry.mol_obj

    atm_grps_mngr = perceiver.perceive_atom_groups(nb_compounds, mol_objs_dict=mol_objs_dict)

    return atm_grps_mngr
    
#TODO: allow this to change
def create_ifp(atm_grps_mngr, luna_params):
    ifp_num_levels = luna_params["ifp_num_levels"]
    ifp_radius_step = luna_params["ifp_radius_step"]
    ifp_diff_comp_classes = luna_params["ifp_diff_comp_classes"]
    ifp_type = luna_params["ifp_type"]
    ifp_count =  luna_params["ifp_count"]
    ifp_length = luna_params["ifp_length"]
    
    sg = ShellGenerator(ifp_num_levels, ifp_radius_step,
                        diff_comp_classes=ifp_diff_comp_classes,
                        ifp_type=ifp_type)
    sm = sg.create_shells(atm_grps_mngr)

    unique_shells = not ifp_count
    
    return sm.to_fingerprint(fold_to_length = ifp_length, 
                      unique_shells = unique_shells, 
                      count_fp = ifp_count)
