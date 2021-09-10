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
import matplotlib.pyplot as plt

from torchvision.transforms.functional import pad
from torchvision import transforms
import numpy as np
import numbers
import csv
import PIL



#Access to the dataset of images and the label
class MahanArtDataset(Dataset):
	def __init__(self, dataframe, transform=None):
		self.data = dataframe
		self.data['classification'] = self.data['classification'].astype('category').cat.codes
		self.transform = transform
		self.len = len(self.data)
		self.show = False

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
		
		if (self.show):
			plt.figure()
			plt.imshow(image) 
			plt.show()  # display it
				
		#Preprocessing transforms
		if self.transform:
			#randomCrop expects the last two dimensions to be H and W
			image = np.transpose(image, (2,0,1))
			image = torch.from_numpy(image)
			
			image = self.transform(image)
			
			#image = image.numpy()
			#transpose back into standard H W RGB format
			image = np.transpose(image, (1,2,0))
			
			if (self.show):
				plt.imshow(image) 
				plt.show() 
			
		return {'image':image, 'classification':classification}

#custom transform
#given a set of images of arbitrary size, we find the largest dimension and format/pad the
# images to fit the model size
class SquarePad(object):
	def __init__(self, csvpath, fill=0, padding_mode='constant'):
		assert isinstance(fill, (numbers.Number, str, tuple))
		assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

		self.fill = fill
		self.padding_mode = padding_mode
		self.largest_dim = self.largest_frame_size(csvpath)
		
	def get_padding(self, image, max_wh):
		max_wh = max_wh[0]     
		w, h = image.shape[2], image.shape[1]
		hp = (int) (max_wh - w)
		vp = (int) (max_wh - h)
		return (hp, 0, vp, 0)
		
	#Assumes write_art_labels_to_csv() was run
	def largest_frame_size(self, csvpath):
		width = 0
		height = 0
		#open file
		with open(csvpath, 'r') as csv_file:
			next(csv_file)
			row_reader = csv.reader(csv_file)
			#run through each image
			for row in row_reader:
				#read in image
				image = PIL.Image.open(row[1])
				w_t, h_t = image.size
				if (w_t > width):
					width = w_t
				if (h_t > height):
					height = h_t
		return (width, height)
        
	def __call__(self, img):
		"""
		Args:
		    img (PIL Image): Image to be padded.

		Returns:
		    PIL Image: Padded image.
		"""
		return F.pad(img, self.get_padding(img, self.largest_dim), self.padding_mode, self.fill)

	def __repr__(self):
		return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
			format(self.fill, self.padding_mode)

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
		

