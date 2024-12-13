###config values for file uploads
INPUTS_FOLDER = 'inputs'
UPLOAD_FOLDER = 'outputs'
DEFAULT_FOLDER = 'defaults'
DEFAULT_PARTY_CONFIG = f'{DEFAULT_FOLDER}/party_default.conf'
MOL_BATCH_SIZE = 50 # batch size for interaction calculations
MOL_PAGE_SIZE = 12 # how many molecules to dispay per page
MIN_MOL_QUEUE_SIZE = 50
MAX_MOL_QUEUE_SIZE = 200

# made a copy of all the default LUNA configs so I'd know where they were
LUNA_default_configs = {
	"LUNA_config": f"{DEFAULT_FOLDER}/LUNA_default.cfg",
	"feat_cfg": f"{DEFAULT_FOLDER}/atom_prop_default.fdef",
	"inter_cfg":f"{DEFAULT_FOLDER}/inter_default.cfg",
	"filter_cfg":f"{DEFAULT_FOLDER}/filter_default.cfg",
	"bind_cfg":f"{DEFAULT_FOLDER}/bind_default.cfg",
}

luna_name_convert = {
	"luna.cfg":"LUNA_config",
	".fdef":"feat_cfg",
	"inter.cfg":"inter_cfg",
	"filter.cfg":"filter_cfg",
	"bind.cfg":"bind_cfg"
}

party_config_options = {
	'output_type': ['ordinal', 'classes'],
	'uncertainty_method': ['ensemble', 'dropout', 'distance'],
	'data_split': ['bootstrap', 'full_split'],
	'distance_method': ['tanimoto', 'euclidean']
}

party_config_types = {
	'learning_rate': float,
	'n_neurons': int,
	'hidden_layers': int,
	'weight_decay': float,
	'dropout': float,
	'output_options': list,
	'output_type': str,
	'uncertainty_method':str,
	'retrain_frequency': int,
	'max_epochs': int,
	'patience': int,
	'committee_size': int,
	'data_split': str,
	'passes': int,
	'distance_method': str,
	'kNN':int
}