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

#needed for write_art_labels_to_csv()
import os
import csv

from classes import MahanArtDataset

TRAIN_TEST_RATIO = 0.8
VALIDATION_TRAIN_RATIO = 0.1

SEED = 1234

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

#Assumes write_art_labels_to_csv() was run
def largest_frame_size(csvpath):
	width = 100000
	height = 1000000
	#open file
	with open(csvpath, 'r') as csv_file:
		next(csv_file)
		row_reader = csv.reader(csv_file)
		#run through each image
		for row in row_reader:
			#read in image
			image = PIL.Image.open(row[1])
			w_t, h_t = image.size
			if (w_t < width):
				width = w_t
			if (h_t < height):
				height = h_t
	return (width, height)

#Define transforms on the data, collect the data into torch iterators, instantiate model object
def prepare_data(csvpath, frame_size):

	#Determine what preprocessing steps the photos should go through
	#	ToPILImage() is so that ToTensor() doesn't complain
	chosen_transforms = transforms.Compose([	#transforms.Resize(size=1600),
							transforms.RandomCrop(frame_size, padding = 2),
							transforms.ToPILImage(), 
							transforms.ToTensor()])

	#Split train/val/test from pandas dataframe
	origin_df = pd.read_csv(csvpath)
	train_df = origin_df.sample(frac=TRAIN_TEST_RATIO, random_state=SEED)
	test_df = origin_df.drop(train_df.index)
	val_df = train_df.sample(frac=VALIDATION_TRAIN_RATIO, random_state=SEED)
	train_df = train_df.drop(val_df.index)
	
	#Load data references into memory
	training_data = MahanArtDataset(train_df, transform=chosen_transforms)
	validation_data = MahanArtDataset(val_df, transform=chosen_transforms)
	testing_data = MahanArtDataset(test_df, transform=chosen_transforms)
	
	#print(f'Number of initial training examples: {len(training_data)}')
	#print(f'Number of initial validation examples: {len(validation_data)}')
	#print(f'Number of initial testing examples: {len(testing_data)}')

	#Define iterators from pytorch, helps manage the data references
	BATCH_SIZE = 115
	train_iterator = data.DataLoader(training_data, shuffle = True, batch_size = BATCH_SIZE)
	valid_iterator = data.DataLoader(validation_data, batch_size = BATCH_SIZE)
	test_iterator = data.DataLoader(testing_data, batch_size = BATCH_SIZE)

	#print("Attempting to access classifications")
	#for item in train_iterator:
	#	print("image is: " + str(item['classification']))
	#print("Classification access complete")

	return train_iterator, valid_iterator, test_iterator

#Written by Ben Trevett
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
    
#Written by Ben Trevett
def train(model, iterator, optimizer, criterion, device):
	epoch_loss = 0
	epoch_acc = 0
	model.train()
	for item in iterator:
		x=item['image'].to(device)
		y=item['classification'].to(device)
		optimizer.zero_grad()
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
def train_model(NUM_EPOCHS, model, train_iterator, valid_iterator, output_filename):

	#Look at computer hardware
	optimizer = optim.Adam(model.parameters())
	criterion = nn.CrossEntropyLoss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	criterion = criterion.to(device)
	
	best_valid_loss = float('inf')
	
	for epoch in range(NUM_EPOCHS):
		start_time = time.monotonic()
		
		train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
		valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			torch.save(model.state_dict(), output_filename)
		    
		end_time = time.monotonic()
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)

		print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
		print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

#Find out what the accuracy is on test data
def test_model(output_filename, model, test_iterator):
	model.load_state_dict(torch.load(output_filename))

	criterion = nn.CrossEntropyLoss()
	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = 'cpu'
	model = model.to(device)
	#criterion = criterion.to(device)
	
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
	
	
	
