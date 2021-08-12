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
import matplotlib.pyplot as plt
import numpy as np

#import copy
import random
import time

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from sklearn.model_selection import train_test_split

#needed for write_art_labels_to_csv()
import os
import csv

from classes import MahanArtDataset


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

	for directory in os.listdir(datapath):
		classpath = datapath+"/"+directory
		for filename in os.listdir(classpath):
			filepath = classpath+"/"+filename
			row = [filename, filepath, directory]
			writer.writerow(row)
	csv_file.close()




def train_model(csvpath):

	#Determine what preprocessing steps the photos should go through
	chosen_transforms = transforms.Compose([	transforms.RandomCrop(28, padding = 2),
							transforms.ToTensor()])

	#Load data references into memory
	#train_data = MahanArtDataset(csvpath, train = True, transform=chosen_transforms)
	#test_data = MahanArtDataset(csvpath, train = False, transform=chosen_transforms)

	train_data, test_data = MahanArtDataset(csvpath, transform=chosen_transforms)

	#Split off validation data from train data
	VALID_RATIO = 0.9
	n_train_examples = int(len(train_data) * VALID_RATIO)
	n_valid_examples = len(train_data) - n_train_examples
	train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
		                                   
	print(f'Number of initial training examples: {len(train_data)}')
	print(f'Number of initial validation examples: {len(valid_data)}')
	print(f'Number of initial testing examples: {len(test_data)}')

	#Define iterators from pytorch, helps manage the data references
	BATCH_SIZE = 64
	train_iterator = data.DataLoader(train_data, shuffle = True, batch_size = BATCH_SIZE)
	valid_iterator = data.DataLoader(valid_data, batch_size = BATCH_SIZE)
	test_iterator = data.DataLoader(test_data, batch_size = BATCH_SIZE)


	'''
	class MLP(nn.Module):
	    def __init__(self, input_dim, output_dim):
		super().__init__()
		        
		self.input_fc = nn.Linear(input_dim, 250)
		self.hidden_fc = nn.Linear(250, 100)
		self.output_fc = nn.Linear(100, output_dim)
		
	    def forward(self, x):
		
		#x = [batch size, height, width]
		
		batch_size = x.shape[0]

		x = x.view(batch_size, -1)
		
		#x = [batch size, height * width]
		
		h_1 = F.relu(self.input_fc(x))
		
		#h_1 = [batch size, 250]

		h_2 = F.relu(self.hidden_fc(h_1))

		#h_2 = [batch size, 100]

		y_pred = self.output_fc(h_2)
		
		#y_pred = [batch size, output dim]
		
		return y_pred, h_2



	INPUT_DIM = 28 * 28 * 3
	OUTPUT_DIM = 7

	model = MLP(INPUT_DIM, OUTPUT_DIM)

	def count_parameters(model):
	    return sum(p.numel() for p in model.parameters() if p.requires_grad)
	    
	print(f'The model has {count_parameters(model):,} trainable parameters')

	optimizer = optim.Adam(model.parameters())

	criterion = nn.CrossEntropyLoss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	criterion = criterion.to(device)
	def calculate_accuracy(y_pred, y):
	    top_pred = y_pred.argmax(1, keepdim = True)
	    correct = top_pred.eq(y.view_as(top_pred)).sum()
	    acc = correct.float() / y.shape[0]
	    return acc
	    


	def train(model, iterator, optimizer, criterion, device):
	    
	    epoch_loss = 0
	    epoch_acc = 0
	    
	    model.train()
	    
	    for item in iterator:
	    	print(item['classification'])
	    
	    for (x, y) in iterator:
		
		x = x.to(device)
		y = y.to(device)
		
		optimizer.zero_grad()
		        
		y_pred, _ = model(x)
		
		loss = criterion(y_pred, y)
		
		acc = calculate_accuracy(y_pred, y)
		
		loss.backward()
		
		optimizer.step()
		
		epoch_loss += loss.item()
		epoch_acc += acc.item()
		
	    return epoch_loss / len(iterator), epoch_acc / len(iterator)

	def evaluate(model, iterator, criterion, device):
	    
	    epoch_loss = 0
	    epoch_acc = 0
	    
	    model.eval()
	    
	    with torch.no_grad():
		
		for (x, y) in iterator:

		    x = x.to(device)
		    y = y.to(device)

		    y_pred, _ = model(x)

		    loss = criterion(y_pred, y)

		    acc = calculate_accuracy(y_pred, y)

		    epoch_loss += loss.item()
		    epoch_acc += acc.item()
		
	    return epoch_loss / len(iterator), epoch_acc / len(iterator)
	    
	def epoch_time(start_time, end_time):
	    elapsed_time = end_time - start_time
	    elapsed_mins = int(elapsed_time / 60)
	    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	    return elapsed_mins, elapsed_secs
	    


	EPOCHS = 10

	best_valid_loss = float('inf')

	for epoch in range(EPOCHS):
	    
	    start_time = time.monotonic()
	    
	    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
	    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
	    
	    if valid_loss < best_valid_loss:
		best_valid_loss = valid_loss
		torch.save(model.state_dict(), 'tut1-model.pt')
	    
	    end_time = time.monotonic()

	    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
	    
	    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
	    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
	    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

	model.load_state_dict(torch.load('tut1-model.pt'))

	test_loss, test_acc = evaluate(model, test_iterator, criterion, device)


	print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
	'''
