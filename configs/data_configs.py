from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    'animalfaces_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['animalfaces-train'],
		'train_target_root': dataset_paths['animalfaces-train'],
		'test_source_root': dataset_paths['animalfaces-test'],
		'test_target_root': dataset_paths['animalfaces-test'],
	},
    'flowers_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['flowers-train'],
		'train_target_root': dataset_paths['flowers-train'],
		'test_source_root': dataset_paths['flowers-test'],
		'test_target_root': dataset_paths['flowers-test'],
	},
    'vggface_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['vggface-train'],
		'train_target_root': dataset_paths['vggface-train'],
		'test_source_root': dataset_paths['vggface-test'],
		'test_target_root': dataset_paths['vggface-test'],
	},
	'af_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['af_train'],
		'train_target_root': dataset_paths['af_train'],
		'valid_source_root': dataset_paths['af_valid'],
		'valid_target_root': dataset_paths['af_valid'],
	},
	'fl_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['fl_train'],
		'train_target_root': dataset_paths['fl_train'],
		'valid_source_root': dataset_paths['fl_valid'],
		'valid_target_root': dataset_paths['fl_valid'],
	},
}
