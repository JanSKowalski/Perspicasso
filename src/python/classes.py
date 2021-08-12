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
#import matplotlib.pyplot as plt
import numpy as np

#import copy
import random
import time

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from sklearn.model_selection import train_test_split

#Access to the dataset of images and the label
class MahanArtDataset(Dataset):

	def __init__(self, csv_file, train=True, transform=None):
		self.data = pd.read_csv(csv_file)
		self.data['classification'] = self.data['classification'].astype('category').cat.codes
		self.transform = transform
		#Split train/test
		msk = np.random.rand(len(self.data)) < 0.8
		train_data = self.data[msk]
		test_data = self.data[~msk]
		if (train):
			self.data=train_data
		else:
			self.data=test_data
		self.len = len(self.data)

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		imagepath = self.data['filepath'][index]
		image = io.imread(imagepath)
		#image = np.array(image)
		classification = self.data['classification'][index]
		
		#preprocessing transforms
		if self.transform:
			image = torch.from_numpy(image)
			#randomCrop expects the last two dimensions to be H and W
			image = np.transpose(image, (2,0,1))
			image = self.transform(image)
			#transpose back into standard H W RGB format
			image = np.transpose(image, (1,2,0))

		sample = {'image':image, 'classification':classification}
		return sample


