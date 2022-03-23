from argparse import ArgumentParser


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--dataset_type', default='af_encode', type=str, help='Type of dataset/experiment to run')
		self.parser.add_argument('--psp_checkpoint_path', default='/data/dgq/fpsp/pretrained_models/psp_iteration_90000.pt', type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--checkpoint_path', default='experiment/logs/log/checkpoints/iteration_40000.pt', type=str, help='Path to AGE model checkpoint')
		self.parser.add_argument('--n_distribution_path', type=str, default='data/n_distribution.npy', help='Path to n distribution')
		self.parser.add_argument('--train_data_path', type=str, default='/data/dgq/fpsp/data/animal_faces/train_without_cate', help='Path to directory of training set')
		self.parser.add_argument('--test_data_path', type=str, default='/data/dgq/fpsp/data/animal_faces/test', help='Path to to directory of inference inputs')
		self.parser.add_argument('--output_path', type=str, default='data/output', help='Path to save outputs')
		self.parser.add_argument('--class_embedding_path', type=str, default='data/', help='Path to save class embeddings')
		
		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')
		
		self.parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')
		self.parser.add_argument('--A_length', default=100, type=int, help='Length of A')
		self.parser.add_argument('--alpha', default=1, type=int, help='Editing intensity alpha')
		self.parser.add_argument('--beta', default=0.005, type=int, help='Direction selection threshold in A')
		self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs to 256x256 or keep at 1024x1024')
	def parse(self):
		opts = self.parser.parse_args()
		return opts