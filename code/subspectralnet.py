"""
This script contains the heart of the code: the SubSpectralNet architecture.

Updated February 2019
Sai Samarth R Phaye
"""

# PyTorch
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils

# Math
import math

class SubSpectralNet(nn.Module):
	""" SubSpectralNet architecture """
	def __init__(self, sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins, use_gpu):
		"""
		Init the model layers
		
		Parameters
		----------
		sub_spectrogram_size : int
			Size of the SubSpectrogram. Default: 20
		
		sub_spectrogram_mel_hop : int
			Mel-bin hop size of the SubSpectrogram. Default 10

		n_mel_bins : int
			Number of mel-bins of the Spectrogram extracted. Default: 40.

		use_gpu : Bool
			Use GPU or not. Default: True
		"""
		super(SubSpectralNet, self).__init__()
		self.sub_spectrogram_size, self.sub_spectrogram_mel_hop, self.n_mel_bins, self.use_gpu = sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins, use_gpu

		# For max-pool after the second conv-layer
		self.n_max_pool = int(self.sub_spectrogram_size / 10)

		# Number of SubSpectrograms: used for defining the number of conv-layers
		self.n_sub_spectrograms = 0

		while(self.sub_spectrogram_mel_hop*self.n_sub_spectrograms <= self.n_mel_bins - self.sub_spectrogram_size):
			self.n_sub_spectrograms = self.n_sub_spectrograms + 1

		# init the layers
		self.conv1 = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3) for _ in range(self.n_sub_spectrograms)])
		self.conv1_bn = nn.ModuleList([nn.BatchNorm2d(32) for _ in range(self.n_sub_spectrograms)])
		self.mp1 = nn.ModuleList([nn.MaxPool2d((self.n_max_pool,5)) for _ in range(self.n_sub_spectrograms)])
		self.drop1 = nn.ModuleList([nn.Dropout(0.3) for _ in range(self.n_sub_spectrograms)])
		self.conv2 = nn.ModuleList([nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3) for _ in range(self.n_sub_spectrograms)])
		self.conv2_bn = nn.ModuleList([nn.BatchNorm2d(64) for _ in range(self.n_sub_spectrograms)])
		self.mp2 = nn.ModuleList([nn.MaxPool2d((4,100)) for _ in range(self.n_sub_spectrograms)])
		self.drop2 = nn.ModuleList([nn.Dropout(0.3) for _ in range(self.n_sub_spectrograms)])

		self.fc1 = nn.ModuleList([nn.Linear(1*2*64, 32) for _ in range(self.n_sub_spectrograms)])
		self.drop3 = nn.ModuleList([nn.Dropout(0.3) for _ in range(self.n_sub_spectrograms)])
		self.fc2 = nn.ModuleList([nn.Linear(32, 10) for _ in range(self.n_sub_spectrograms)])

		numFCs = int(math.log(self.n_sub_spectrograms*32, 2))
		neurons = int(math.pow(2, numFCs))

		self.fcGlobal = []
		tempNeurons =int(32*self.n_sub_spectrograms)
		while(neurons >= 64):
			self.fcGlobal.append(nn.Linear(tempNeurons, neurons))
			self.fcGlobal.append(nn.ReLU(0.3))
			self.fcGlobal.append(nn.Dropout(0.3))
			tempNeurons = neurons
			neurons = int(neurons / 2)
		self.fcGlobal.append(nn.Linear(tempNeurons, 10))		
		self.fcGlobal = nn.ModuleList(self.fcGlobal)

	def forward(self, x):
		"""
		Feed-forward pass

		Parameters
		----------
		x : tensor
			Input batch. Size: [batch_size, channels, sub_spectrogram_size, n_time_indices, n_sub_spectrograms]. Default [16, 1, 20, 500, 3]

		Returns
		-------
		outputs: tensor
			final output of the model. Size: [batch_size, n_sub_spectrograms, n_labels]. Default: [16, 4, 10]
		"""
		outputs = []
		intermediate = []
		x = x.float() 
		if self.use_gpu:
			x = x.cuda()
		input_var = x

		# for every sub-spectrogram
		for i in range(x.shape[4]):
			x = input_var
			x = self.conv1[i](x[:, :, :, :, i])
			x = self.conv1_bn[i](x)
			x = F.relu(x)
			x = self.mp1[i](x)
			x = self.drop1[i](x)
			x = self.conv2[i](x)
			x = self.conv2_bn[i](x)
			x = F.relu(x)
			x = self.mp2[i](x)
			x = self.drop2[i](x)
			x = x.view(-1, 1*2*64)
			x = self.fc1[i](x)
			x = F.relu(x)
			intermediate.append(x)
			x = self.drop3[i](x)
			x = self.fc2[i](x)
			x = x.view(-1, 1, 10)
			outputs.append(x)

		# extracted intermediate layers
		x = torch.cat((intermediate), 1)

		# global classification
		for i in range(len(self.fcGlobal)):
			x = self.fcGlobal[i](x)
		x = x.view(-1, 1, 10)
		outputs.append(x)

		# all the outputs (low, mid and high band + global classifier)
		outputs = torch.cat((outputs), 1)
		outputs = F.log_softmax(outputs, dim=2)
		return outputs
