"""
This script contains the basic building blocks of the DataLoader and Transforms for preprocessing the data in batches.

Updated February 2019
Sai Samarth R Phaye
"""

# Basics
import numpy as np
import os

# Audio Processing
import librosa
import dcase_util

# PyTorch
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from ignite._utils import convert_tensor

# Tensorboard
try:
	from tensorboardX import SummaryWriter
except ImportError:
	raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

class ToTensor(object):
	""" Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		data, label = sample['data'], sample['label']

		# swap color axis (not required) 
		data = data.transpose((0, 1, 2, 3))

		return {'data': torch.from_numpy(data),
				'label': torch.from_numpy(label)}



class Normalize(object):
	"""Bin-wise Normalization of the Mel-Spectrograms."""

	def __init__(self):
		# Use the pre-calculated mean and standard deviations
		self.mean = np.load('mean_final.npy')
		self.std = np.load('std_final.npy')
		self.mean = torch.from_numpy(self.mean)
		self.std = torch.from_numpy(self.std)
		self.mean = torch.reshape(self.mean, [40,1])
		self.std = torch.reshape(self.std, [40, 1])

	def __call__(self, sample):
		data, label = sample['data'], sample['label']

		data[:,:,:,0] = (data[:,:,:,0] - self.mean)/self.std

		return {'data': data,
				'label': label}

class ToSubSpectrograms(object):
	""" Generate Sub-Spectrogram Tensors """
	def __init__(self, sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins):
		"""
		Parameters
		----------
		sub_spectrogram_size : int
			Size of the SubSpectrogram. Default: 20

		sub_spectrogram_mel_hop : int
			Mel-bin hop size of the SubSpectrogram. Default 10

		n_mel_bins : int
			Number of mel-bins of the Spectrogram extracted. Default: 40.
		"""
		self.sub_spectrogram_size, self.sub_spectrogram_mel_hop, self.n_mel_bins = sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins
		
	def __call__(self, sample):
		"""
		Parameters
		----------
		sample : PyTorch tensor
			The input tensor data and label
		Returns
		-------
		sub_spectrograms: tensor
			A list of sub-spectrograms. Default size [channels, sub_spectrogram_size, time_indices, n_sub_spectrograms]
		label: tensor
			Corresponding label
		"""
		spectrogram, label = sample['data'], sample['label']

		i = 0
		sub_spectrograms = torch.from_numpy(np.asarray([]))
		while(self.sub_spectrogram_mel_hop*i <= self.n_mel_bins - self.sub_spectrogram_size):

			# Extract a Sub-Spectrogram
			subspectrogram = spectrogram[:,i*self.sub_spectrogram_mel_hop:i*self.sub_spectrogram_mel_hop+self.sub_spectrogram_size,:, :]

			if i == 0:
				sub_spectrograms = subspectrogram
			else:
				sub_spectrograms = torch.cat((subspectrogram, sub_spectrograms), 3)

			i = i + 1

		return sub_spectrograms, label

class DCASEDataset(Dataset):
	""" DCASE 2018 Dataset extraction """

	def __init__(self, csv_file, root_dir, transform=None):
		"""
		Parameters
		----------
		csv_file : str
			Location of the CSV file, with respect to the root_dir (should be something like 'evaluation_setup/fold1_train.txt')

		root_dir : int
			Root directory of the dataset folder (which contains 'audio' and 'evaluation_setup' folders).

		transform : PyTorch transforms, optional
			Used for transforming the data
		"""

		list1 = []
		list2 = []
		with open(root_dir + csv_file, 'r') as f:
			content = f.readlines()
			for x in content:
				row = x.split()
				list1.append(row[0])
				list2.append(row[1])
		self.root_dir = root_dir
		self.transform = transform
		self.datalist = list1
		self.labels = list2
		self.default_labels = ['airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']

	def __len__(self):
		""" set the len(object) funciton """
		return len(self.datalist)

	def __getitem__(self, idx):
		"""
		Function to extract the spectrogram samples and labels from the audio dataset.
		"""
		wav_name = os.path.join(self.root_dir,
								self.datalist[idx])

		# extracting with 22050 sampling rate by default
		audioContainer = dcase_util.containers.AudioContainer().load(filename=wav_name, fs=22050)
		# use only one channel (NOTE: In the paper, both channels are used)
		audio = audioContainer.data[0]
		sr = audioContainer.fs

		# extract mel-spectrogram. results in a time-frequency matrix of 40x500 size.
		spec = librosa.feature.melspectrogram(y=audio, sr=sr, S=None, n_fft=883, hop_length=441, n_mels=40)
		logmel = librosa.core.amplitude_to_db(spec)
		logmel = np.reshape(logmel, [1, logmel.shape[0], logmel.shape[1], 1])
		
		label = np.asarray(self.default_labels.index(self.labels[idx]))
		sample = {'data': logmel, 'label': label}
		if self.transform:
			sample = self.transform(sample)
		return sample

def get_data_loaders(train_batch_size, test_batch_size, sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins, use_cuda, root_dir, train_dir, eval_dir):
	"""
	Function to return the data loaders

	Parameters
	----------
	train_batch_size : int
		Size of the training batch. Default: 16

	test_batch_size : int
		size of the testing batch. Default: 16

	sub_spectrogram_size : int
		Size of the SubSpectrogram. Default 20
		
	sub_spectrogram_mel_hop : int
		Mel-bin hop size of the SubSpectrogram. Default 10

	n_mel_bins : int
		Number of mel-bins of the Spectrogram extracted. Default: 40.

	use_gpu : Bool
		Use GPU or not. Default: True

	root_dir : str
		Directory of the folder which contains the dataset (has 'audio' and 'evaluation_setup' folders inside)

	train_dir : str
		Set as default: 'evaluation_setup/train_fold1.txt'

	eval_dir : str
		Set as default: 'evaluation_setup/evaluate_fold1.txt'

	Returns
	-------
	train_loader and val_loader
		data loading objects
	"""

	kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

	data_transform = transforms.Compose([
		ToTensor(), Normalize(), ToSubSpectrograms(sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins)
		])

	dcase_train = DCASEDataset(csv_file=train_dir,
								root_dir=root_dir, transform=data_transform)
	dcase_test = DCASEDataset(csv_file=eval_dir,
								root_dir=root_dir, transform=data_transform)

	train_loader = torch.utils.data.DataLoader(dcase_train,
			batch_size=train_batch_size, shuffle=True, **kwargs)

	val_loader = torch.utils.data.DataLoader(dcase_test,
			batch_size=test_batch_size, shuffle=True, **kwargs)

	return train_loader, val_loader

def create_summary_writer(model, data_loader, log_dir):
	"""
	Create the summary writer for TensorBoard

	Parameters
	----------
	model : PyTorch model object
		Size of the training batch. 

	data_loader : data_loader
		Data loader object to create the graph

	log_dir : str
		Directory to save the logs

	Returns
	-------
	train_loader and val_loader
		data loading objects
	"""
	writer = SummaryWriter(log_dir=log_dir)
	data_loader_iter = iter(data_loader)
	x, y = next(data_loader_iter)
	try:
		writer.add_graph(model, x)
	except Exception as e:
		print("Failed to save model graph: {}".format(e))
	return writer

def prepare_batch(batch, device=None, non_blocking=False):
	"""
	Inbuilt function in the ignite._utils, for converting the data to tensors.
	Returns the tensors of the input data, using convert_tensor function.
	"""
	x, y = batch
	return (convert_tensor(x, device=device, non_blocking=non_blocking),
		convert_tensor(y, device=device, non_blocking=non_blocking))
