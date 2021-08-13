import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

import random
import time

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from sklearn.model_selection import train_test_split


#Access to the dataset of images and the label
class MahanArtDataset(Dataset):
	def __init__(self, dataframe, transform=None):
		self.data = dataframe
		self.data['classification'] = self.data['classification'].astype('category').cat.codes
		self.transform = transform
		self.len = len(self.data)

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		#Lookup data
		imagepath = self.data['filepath'].iloc[index]
		image = io.imread(imagepath)
		#torch long type required by CrossEntropyLoss()
		#https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
		classification = torch.tensor(self.data['classification'].iloc[index], dtype=torch.long)
				
		#Quick and dirty greyscale to rgb conversion
		if (len(image.shape)==2):
			image = np.array([image, image, image])
			image = np.transpose(image, (1,2,0))
				
		#Preprocessing transforms
		if self.transform:
			#randomCrop expects the last two dimensions to be H and W
			image = np.transpose(image, (2,0,1))
			image = torch.from_numpy(image)
			image = self.transform(image)
			#image = image.numpy()
			#transpose back into standard H W RGB format
			image = np.transpose(image, (1,2,0))
			
		return {'image':image, 'classification':classification}

class MLP(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()

		self.input_fc = nn.Linear(input_dim, 250)
		self.hidden_fc = nn.Linear(250, 100)
		self.output_fc = nn.Linear(100, output_dim)

	def forward(self, x):
		batch_size = x.shape[0]
		x = x.view(batch_size, -1)
		h_1 = F.relu(self.input_fc(x))
		h_2 = F.relu(self.hidden_fc(h_1))
		y_pred = self.output_fc(h_2)
		return y_pred, h_2
	
	def name(self):
		return "MLP_neural_network"
