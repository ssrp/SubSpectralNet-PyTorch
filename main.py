"""
This script runs the SubSpectralNet training in PyTorch.

Updated February 2019
Sai Samarth R Phaye
"""

from __future__ import print_function, division

# Basics
import argparse
import os
import numpy as np

from subspectralnet import *

from utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Ignite Framework
import torch
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

def run(train_batch_size, test_batch_size, epochs, lr, log_interval, log_dir, no_cuda, sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins, seed, root_dir, train_dir, eval_dir):
	"""
	Model runner

	Parameters
	----------
	train_batch_size : int
		Size of the training batch. Default: 16

	test_batch_size : int
		size of the testing batch. Default: 16

	epochs : int
		Number of training epochs. Default: 200

	lr : float
		Learning rate for the ADAM optimizer. Default: 0.001

	log_interval : int
		Interval for logging data: Default: 10

	log_dir : str
		Directory to save the logs

	no_cuda : Bool
		Should you NOT use cuda? Default: False

	sub_spectrogram_size : int
		Size of the SubSpectrogram. Default 20
		
	sub_spectrogram_mel_hop : int
		Mel-bin hop size of the SubSpectrogram. Default 10

	n_mel_bins : int
		Number of mel-bins of the Spectrogram extracted. Default: 40.

	seed : int
		Torch random seed value, for reproducable results. Default: 1

	root_dir : str
		Directory of the folder which contains the dataset (has 'audio' and 'evaluation_setup' folders inside)

	train_dir : str
		Set as default: 'evaluation_setup/train_fold1.txt'

	eval_dir : str
		Set as default: 'evaluation_setup/evaluate_fold1.txt'
	"""

	# check if possible to use CUDA
	use_cuda = not no_cuda and torch.cuda.is_available()

	# set seed
	torch.manual_seed(seed)

	# Map to GPU
	device = torch.device("cuda" if use_cuda else "cpu")

	# Load the data loaders
	train_loader, val_loader = get_data_loaders(train_batch_size, test_batch_size, sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins, use_cuda, root_dir, train_dir, eval_dir)

	# Get the model
	model = SubSpectralNet(sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins, use_cuda).to(device)

	# Init the TensorBoard summary writer
	writer = create_summary_writer(model, train_loader, log_dir)

	# Init the optimizer
	optimizer = optim.Adam(model.parameters(), lr=lr)

	# Use GPU if possible
	if device:
		model.to(device)

	def update_model(engine, batch):
		"""Prepare batch for training: pass to a device with options.

		"""
		model.train()
		optimizer.zero_grad()
		
		inputs, label = prepare_batch(batch, device=device)
		output = model(inputs)
		losses = []
		for ite in range(output.shape[1]):
			losses.append(F.nll_loss(output[:,ite,:], label))
		loss = sum(losses)
		loss.backward()
		optimizer.step()
		return losses, output

	# get the trainer module
	trainer = Engine(update_model)

	def evaluate(engine, batch):
		"""Prepare batch for training: pass to a device with options.
		"""
		model.eval()
		with torch.no_grad():
			inputs, label = prepare_batch(batch, device=device)
			output = model(inputs)
			losses = []
			correct = []
			for ite in range(output.shape[1]):
				losses.append(F.nll_loss(output[:,ite,:], label, reduction='sum').item())
		return losses, output, label

	# get the evaluator module
	evaluator = Engine(evaluate)

	# define output transforms for multiple outputs.
	def output_transform1(output):
		# `output` variable is returned by above `process_function`
		losses, correct, label = output
		return correct[:,0,:], label

	metric = Accuracy(output_transform=output_transform1)
	metric.attach(evaluator, "acc_highband")
	metric = Loss(F.nll_loss, output_transform=output_transform1)
	metric.attach(evaluator, "loss_highband")

	def output_transform2(output):
		# `output` variable is returned by above `process_function`
		losses, correct, label = output
		return correct[:,1,:], label

	metric = Accuracy(output_transform=output_transform2)
	metric.attach(evaluator, "acc_midband")
	metric = Loss(F.nll_loss, output_transform=output_transform2)
	metric.attach(evaluator, "loss_midband")

	def output_transform3(output):
		# `output` variable is returned by above `process_function`
		losses, correct, label = output
		return correct[:,2,:], label

	metric = Accuracy(output_transform=output_transform3)
	metric.attach(evaluator, "acc_lowband")
	metric = Loss(F.nll_loss, output_transform=output_transform3)
	metric.attach(evaluator, "loss_lowband")

	def output_transform(output):
		# `output` variable is returned by above `process_function`
		losses, correct, label = output
		return correct[:,3,:], label

	metric = Accuracy(output_transform=output_transform)
	metric.attach(evaluator, "acc_globalclassifier")
	metric = Loss(F.nll_loss, output_transform=output_transform)
	metric.attach(evaluator, "loss_globalclassifier")

	# Log the events in Ignite: EVERY ITERATION
	@trainer.on(Events.ITERATION_COMPLETED)
	def log_training_loss(engine):
		iter = (engine.state.iteration - 1) % len(train_loader) + 1
		if iter % log_interval == 0:
			losses, output = engine.state.output
			epoch = engine.state.epoch
			print('Train Epoch: {} [{}/{}]\tLosses: {:.6f} (Top Band), {:.6f} (Mid Band), {:.6f} (Low Band), {:.6f} (Global Classifier)'.format(
				epoch, iter, len(train_loader), losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()))
			# TensorBoard Logs
			writer.add_scalar("training/loss_topband_itr", losses[0].item(), engine.state.iteration)
			writer.add_scalar("training/loss_midband_itr", losses[1].item(), engine.state.iteration)
			writer.add_scalar("training/loss_lowband_itr", losses[2].item(), engine.state.iteration)
			writer.add_scalar("training/loss_global_itr", losses[3].item(), engine.state.iteration)


	# Log the events in Ignite: Test the training data on EVERY EPOCH
	@trainer.on(Events.EPOCH_COMPLETED)
	def log_training_results(engine):
		evaluator.run(train_loader)
		print("Training Results - Epoch: {}  Global accuracy: {:.2f} Avg loss: {:.2f}"
			  .format(engine.state.epoch, evaluator.state.metrics['acc_globalclassifier'], evaluator.state.metrics['loss_globalclassifier']))
		# TensorBoard Logs
		writer.add_scalar("training/global_loss", evaluator.state.metrics['loss_globalclassifier'], engine.state.epoch)
		writer.add_scalar("training/lowband_loss", evaluator.state.metrics['loss_lowband'], engine.state.epoch)
		writer.add_scalar("training/midband_loss", evaluator.state.metrics['loss_midband'], engine.state.epoch)
		writer.add_scalar("training/highband_loss", evaluator.state.metrics['loss_highband'], engine.state.epoch)
		writer.add_scalar("training/global_acc", evaluator.state.metrics['acc_globalclassifier'], engine.state.epoch)
		writer.add_scalar("training/lowband_acc", evaluator.state.metrics['acc_lowband'], engine.state.epoch)
		writer.add_scalar("training/midband_acc", evaluator.state.metrics['acc_midband'], engine.state.epoch)
		writer.add_scalar("training/highband_acc", evaluator.state.metrics['acc_highband'], engine.state.epoch)


	# Log the events in Ignite: Test the validation data on EVERY EPOCH
	@trainer.on(Events.EPOCH_COMPLETED)
	def log_validation_results(engine):
		evaluator.run(val_loader)
		print("Validation Results - Epoch: {}  Global accuracy: {:.2f} Avg loss: {:.2f}"
			  .format(engine.state.epoch, evaluator.state.metrics['acc_globalclassifier'], evaluator.state.metrics['loss_globalclassifier']))
		# TensorBoard Logs
		writer.add_scalar("validation/global_loss", evaluator.state.metrics['loss_globalclassifier'], engine.state.epoch)
		writer.add_scalar("validation/lowband_loss", evaluator.state.metrics['loss_lowband'], engine.state.epoch)
		writer.add_scalar("validation/midband_loss", evaluator.state.metrics['loss_midband'], engine.state.epoch)
		writer.add_scalar("validation/highband_loss", evaluator.state.metrics['loss_highband'], engine.state.epoch)
		writer.add_scalar("validation/global_acc", evaluator.state.metrics['acc_globalclassifier'], engine.state.epoch)
		writer.add_scalar("validation/lowband_acc", evaluator.state.metrics['acc_lowband'], engine.state.epoch)
		writer.add_scalar("validation/midband_acc", evaluator.state.metrics['acc_midband'], engine.state.epoch)
		writer.add_scalar("validation/highband_acc", evaluator.state.metrics['acc_highband'], engine.state.epoch)

	# kick everything off
	trainer.run(train_loader, max_epochs=epochs)

	# close the writer
	writer.close()

	# return the model
	return model

def main():
	"""
	Main method. Initializes the parser, default variables and calls the run function. 
	"""
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch code for SubSpectralNets')

	parser.add_argument('--batch-size', type=int, default=16, metavar='BS',
						help='input batch size for training (default: 16)')

	parser.add_argument('--test-batch-size', type=int, default=16, metavar='TBS',
						help='input batch size for testing (default: 16)')

	parser.add_argument('--epochs', type=int, default=200, metavar='E',
						help='number of epochs to train (default: 200)')

	parser.add_argument('--sub-spectrogram-size', type=int, default=20, metavar='SSS',
						help='sub-spectrogram size (default: 16)')

	parser.add_argument('--sub-spectrogram-mel-hop', type=int, default=10, metavar='MH',
						help='sub-spectrogram mel-hop value (default: 10)')

	parser.add_argument('--time-indices', type=int, default=500, metavar='T',
						help='temporal dimension size of the spectrogram')

	parser.add_argument('--channels', type=int, default=1, metavar='C',
						help='number of audio channels (default: 1)')

	parser.add_argument('--mel-bins', type=int, default=40, metavar='MB',
						help='number of audio channels (default: 40)')

	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.001)')

	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')

	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')

	parser.add_argument('--log-interval', type=int, default=10, metavar='I',
						help='how many batches to wait before logging training status')

	parser.add_argument('--save-model', action='store_true', default=False,
						help='For Saving the current Model')

	parser.add_argument('--root-dir', type=str, default="../../TUT-urban-acoustic-scenes-2018-development/", metavar='RD',
						help='Root directory for the dataset: must contain \'audio\' folder')

	parser.add_argument('--train-dir', type=str, default="evaluation_setup/fold1_train.txt", metavar='TD',
						help='Link to train data labels file')

	parser.add_argument('--eval-dir', type=str, default="evaluation_setup/fold1_evaluate.txt", metavar='ED',
						help='Link to evaluate data labels file')

	parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
						help="log directory for Tensorboard log output")

	args = parser.parse_args()

	# Sub-Spectrogram Size
	sub_spectrogram_size = args.sub_spectrogram_size

	# Init Directories
	root_dir = args.root_dir 
	train_dir = args.train_dir
	eval_dir = args.eval_dir

	# Mel-bins
	n_mel_bins = args.mel_bins

	# Mel-bins sub_spectrogram_mel_hop
	sub_spectrogram_mel_hop = args.sub_spectrogram_mel_hop

	# Time Indices
	timeInd = args.time_indices

	# Channels used
	channels = args.channels

	# get number of classifiers (number of sub-spectrograms + 1 for global classifier)
	numClassifiers = 0
	while(sub_spectrogram_mel_hop*numClassifiers <= n_mel_bins - sub_spectrogram_size):
		numClassifiers = numClassifiers + 1
	# + 1 for global classifier
	numClassifiers = numClassifiers + 1

	# Run the model
	model = run(args.batch_size, args.test_batch_size, args.epochs, args.lr, args.log_interval, args.log_dir, args.no_cuda, sub_spectrogram_size, sub_spectrogram_mel_hop, n_mel_bins, args.seed, root_dir, train_dir, eval_dir)
	
	# save the model
	if (args.save_model):
		torch.save(model.state_dict(),"subspectralnet_cnn.pt")

if __name__ == '__main__':
	"""
	Python main function.
	Calls another main() function because using this main function is too 'main'stream.
	"""
	main()