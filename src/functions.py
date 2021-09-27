import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

#from sklearn import metrics
#from sklearn import decomposition
#from sklearn import manifold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

import random
import time
import PIL

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform

import matplotlib.pyplot as plt
from utils import store_patterns, load_patterns
from visualization import heatmap_grid
import lrp
from lrp.patterns import fit_patternnet, fit_patternnet_positive # PatternNet patterns

#needed for write_art_labels_to_csv()
import os
import csv

from classes import MahanArtDataset	

TRAIN_TEST_RATIO = 0.8
VALIDATION_TRAIN_RATIO = 0.1

SEED = random.randint(1, 1000)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#From Mahan's directory of photos, produce a csv that connects image with label
#Assumes csvpath and datapath are correct, defined in main.py
def write_art_labels_to_csv(datapath, csvpath):
	counter = 0

	#generate csv file with classification
	#id filename 
	csv_file = open(csvpath, 'w')
	writer = csv.writer(csv_file)

	row = ["filename", "filepath", "classification"]
	writer.writerow(row)
	
	size_of_data=0

	for directory in os.listdir(datapath):
		classpath = datapath+"/"+directory
		for filename in os.listdir(classpath):
			filepath = classpath+"/"+filename
			row = [filename, filepath, directory]
			writer.writerow(row)
			size_of_data += 1
	csv_file.close()
	print("There are "+ str(size_of_data) +" data entries in this csv")
	
#Define transforms on the data, collect the data into torch iterators, instantiate model object
def prepare_data(csvpath, frame_size, BATCH_SIZE):
	print(f"SEED: {SEED}")
	initial_transforms = transforms.Compose([transforms.Resize((frame_size, frame_size))])
	
	#Read in data pointers
	origin_df = pd.read_csv(csvpath)
	origin_dataset = MahanArtDataset(origin_df, transform=initial_transforms)
	origin_iterator = data.DataLoader(origin_dataset, batch_size = BATCH_SIZE)
	
	#Determine data statistics for normalization
	means = torch.zeros(3)
	stds = torch.zeros(3)
	for item in origin_iterator:
		img = item["image"]
		means += torch.mean(img, dim = (0, 2, 3))
		stds += torch.std(img, dim = (0, 2, 3))

	means /= len(origin_iterator)
	stds /= len(origin_iterator)
	
	print(means)
	print(stds)
		
	#Split train/val/test from pandas dataframe
	train_df = origin_df.sample(frac=TRAIN_TEST_RATIO, random_state=SEED)
	test_df = origin_df.drop(train_df.index)
	val_df = train_df.sample(frac=VALIDATION_TRAIN_RATIO, random_state=SEED)
	train_df = train_df.drop(val_df.index)
	
	train_transforms = transforms.Compose([
				transforms.Resize((frame_size, frame_size)),
				transforms.RandomRotation(5),
				transforms.RandomHorizontalFlip(0.5),
				transforms.Normalize(mean = means, std = stds)
		])

	test_transforms = transforms.Compose([
		            transforms.Resize((frame_size, frame_size)),
		            transforms.Normalize(mean = means, std = stds)
		        ])
	
	
	#Load data references into memory
	training_data = MahanArtDataset(train_df, transform=train_transforms)
	validation_data = MahanArtDataset(val_df, transform=test_transforms)
	testing_data = MahanArtDataset(test_df, transform=test_transforms)
	
	#Define iterators from pytorch, helps manage the data references
	train_iterator = data.DataLoader(training_data, shuffle = True, batch_size = BATCH_SIZE)
	valid_iterator = data.DataLoader(validation_data, batch_size = BATCH_SIZE)
	test_iterator = data.DataLoader(testing_data, batch_size = BATCH_SIZE)

	return train_iterator, valid_iterator, test_iterator

#Written by Ben Trevett
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
    
#Written by Ben Trevett
def train(model, iterator, optimizer, criterion, device, logging_file):
	epoch_loss = 0
	epoch_acc = 0
	model.train()
	for item in iterator:
		x=item['image'].to(device)
		y=item['classification'].to(device)
		optimizer.zero_grad()
		#print(x)
		#print(y)
		y_pred, _ = model(x)
		loss = criterion(y_pred, y)
		acc = calculate_accuracy(y_pred, y)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
		epoch_acc += acc.item()
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

#Written by Ben Trevett
def evaluate(model, iterator, criterion, device):
	epoch_loss = 0
	epoch_acc = 0
	model.eval()
	with torch.no_grad():
		for item in iterator:
			x=item['image'].to(device)
			y=item['classification'].to(device)
			y_pred, _ = model(x)
			loss = criterion(y_pred, y)
			acc = calculate_accuracy(y_pred, y)
			epoch_loss += loss.item()
			epoch_acc += acc.item()
	
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

#Written by Ben Trevett
def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs

#High level training control -- Important
def train_model(NUM_EPOCHS, model, train_iterator, valid_iterator, output_filename, load_previous):

	logging_file = open("logging.txt", 'a')

	#Look at computer hardware
	optimizer = optim.Adam(model.parameters())
	criterion = nn.CrossEntropyLoss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	start_epoch = 0
	if (load_previous):
		try:
			model, optimizer, start_epoch, criterion = load_checkpoint(model, optimizer, criterion)
			for state in optimizer.state.values():
				for k, v in state.items():
					if isinstance(v, torch.Tensor):
						state[k] = v.to(device)
		except OSError:
			print("Not loading from checkpoint")
			pass
	model = model.to(device)
	criterion = criterion.to(device)
	
	best_valid_loss = float('inf')
	
	#Make sure not to repeat epochs that have already been trained
	for epoch in range(NUM_EPOCHS-start_epoch):
		start_time = time.monotonic()
		
		train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device, logging_file)
		valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
		    
		end_time = time.monotonic()
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)

		print(f'Epoch: {start_epoch+epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
		print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
		logging_file.write(f'Epoch: {start_epoch+epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
		logging_file.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%\n')
		
		logging_file.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%\n')
		
		
		
	state = {'epoch': NUM_EPOCHS, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'criterion': criterion, }
	torch.save(state, output_filename)
	logging_file.close()

#Written by Scott Hawley
#https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/2
# Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
def load_checkpoint(model, optimizer, criterion, filename="MLP_neural_network.pt"):
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        criterion = checkpoint['criterion']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, criterion

#Find out what the accuracy is on test data
def test_model(output_filename, model, test_iterator):
	#model.load_state_dict(torch.load(output_filename))
	optimizer = optim.Adam(model.parameters())
	criterion = nn.CrossEntropyLoss()
	model, a, b, c = load_checkpoint(model, optimizer, criterion, "MLP_neural_network.pt")
	
	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = 'cpu'
	model = model.to(device)
	criterion = criterion.to(device)
	
	test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
	print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
	

	#output confusion matrix
	model.eval()
	with torch.no_grad():
		y_true = []
		y_choice = []
		for item in test_iterator:
			x=item['image'].to(device)
			y=item['classification'].to(device)
			y_pred, _ = model(x)
			for entry in y:
				y_true.append(entry)
			for entry in y_pred:
				y_choice.append(torch.argmax(entry).numpy())
		print("#--------------------------------------#")
		print("Confusion matrix of test predictions:")
		print(confusion_matrix(y_true, y_choice))

	return test_acc
	
	
	
	
	
	
	
##########################################################################
#			LRP stuff	
##########################################################################	
	
def compute_and_plot_explanation(rule, ax_, title=None, postprocess=None, pattern=None, cmap='seismic'): 

	# # # # For the interested reader:
	# This is where the LRP magic happens.
	# Reset gradient
	x.grad = None

	# Forward pass with rule argument to "prepare" the explanation
	y_hat = model.forward(x, explain=True, rule=rule, pattern=pattern)
	# Choose argmax
	y_hat = y_hat[torch.arange(x.shape[0]), y_hat.max(1)[1]]
	# y_hat *= 0.5 * y_hat # to use value of y_hat as starting point
	y_hat = y_hat.sum()

	# Backward pass (compute explanation)
	y_hat.backward()
	attr = x.grad

	if postprocess:  # Used to compute input * gradient
		with torch.no_grad(): 
			attr = postprocess(attr)

	attr = heatmap_grid(attr, cmap_name=cmap)

	if title is None: title = rule
	plot_attribution(attr, ax_, pred, title, cmap=cmap)
	
#Find out what the accuracy is on test data
def test_model_lrp(output_filename, model, test_iterator):
	#model.load_state_dict(torch.load(output_filename))
	optimizer = optim.Adam(model.parameters())
	criterion = nn.CrossEntropyLoss()
	model, a, b, c = load_checkpoint(model, optimizer, criterion, "MLP_neural_network.pt")
	
	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = 'cpu'
	model = model.to(device)
	criterion = criterion.to(device)
	
	test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
	print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
	

	#output confusion matrix
	model.eval()
	with torch.no_grad():
		y_true = []
		y_choice = []
		for item in test_iterator:
			x=item['image'].to(device)
			y=item['classification'].to(device)
			y_pred, _ = model(x)
			for entry in y:
				y_true.append(entry)
			for entry in y_pred:
				y_choice.append(torch.argmax(entry).numpy())
		print("#--------------------------------------#")
		print("Confusion matrix of test predictions:")
		print(confusion_matrix(y_true, y_choice))


	# # # # Patterns for PatternNet and PatternAttribution
	#patterns_all = fit_patternnet(model, train_loader, device=args.device)
	patterns_all = fit_patternnet(model, test_iterator, device)
	store_patterns("pattern_all.pkl", patterns_all)


	pos_patterns_path = (base_path / 'examples' / 'patterns' / 'pattern_pos.pkl').as_posix()
	if not os.path.exists(pos_patterns_path):
		patterns_pos = fit_patternnet_positive(model, train_loader, device)#, max_iter=1)
		store_patterns(pos_patterns_path, patterns_pos)
	else:
		patterns_pos = [torch.from_numpy(p).to(args.device) for p in load_patterns(pos_patterns_path)]


	# # # Plotting
	fig, ax = plt.subplots(2, 5, figsize=(10, 5))

	with torch.no_grad(): 
		x_plot = heatmap_grid(x*2-1, cmap_name="gray")
		plot_attribution(x_plot, ax[0, 0], pred, "Input")

	# compute_and_plot_explanation("gradient", ax[1, 0], title="gradient")
	compute_and_plot_explanation("gradient", ax[1, 0], title="input $\\times$ gradient", postprocess = lambda attribution: attribution * x)

	compute_and_plot_explanation("epsilon", ax[0, 1])
	compute_and_plot_explanation("gamma+epsilon", ax[1, 1])
	# 
	compute_and_plot_explanation("alpha1beta0", ax[0, 2])
	compute_and_plot_explanation("alpha2beta1", ax[1, 2])
	# 
	compute_and_plot_explanation("patternnet", ax[0, 3], pattern=patterns_all, title="PatternNet $S(x)$", cmap='gray')
	compute_and_plot_explanation("patternnet", ax[1, 3], pattern=patterns_pos, title="PatternNet $S(x)_{+-}$", cmap='gray')

	compute_and_plot_explanation("patternattribution", ax[0, 4], pattern=patterns_all, title="PatternAttribution $S(x)$")
	compute_and_plot_explanation("patternattribution", ax[1, 4], pattern=patterns_pos, title="PatternAttribution $S(x)_{+-}$")

	fig.tight_layout()

	fig.savefig((base_path / 'examples' / 'plots' / "mnist_explanations.png").as_posix(), dpi=280)
	plt.show()


	return test_acc	
