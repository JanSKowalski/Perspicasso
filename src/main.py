#! /home/jan/anaconda3/bin/python3

import functions
import csv
import os
import itertools
import functions
from classes import MLP, MLP_w_torchlrp, AlexNet
import subprocess

import os
import sys
import time
import torch
import random
import pathlib
import argparse
import torchvision
import matplotlib.pyplot as plt
#from utils import store_patterns, load_patterns
#from visualization import heatmap_grid
#import lrp
#from lrp.patterns import fit_patternnet, fit_patternnet_positive # PatternNet patterns

from functions import write_results_to_csv



import torch.nn as nn

datapath = "../Dataset/images"
csvpath = "./images.csv"
outputpath = "./plotting.csv"
resultspath = "./data_frame_80"

##########################################################################
##############             Main commands                 #################
##########################################################################
def main():
	frame_size = 80
	for num_epochs in range(10, 60, 10):
		Wall_time_start = time.monotonic()
		for trial_num in range(500):
			simple_MLP_example(trial_num, frame_size, num_epochs)
		Wall_time_end = time.monotonic()
		print("-------------------------------------------------")
		print(f"Frame size: {frame_size}, Number of Epochs: {num_epochs}")
		print(f"Wall Time: %.2f" % (Wall_time_end-Wall_time_start))

##########################################################################

#fixed frame size, fixed number of epochs
def simple_MLP_example(trial_num, frame_size, num_epochs):
	#prepare data as csv
	functions.write_art_labels_to_csv(datapath, csvpath)
	
	#set values for frame size and num epochs
	batch_size = 8
		
	#split train/val/test, then load into data iterators
	#also store the dictionary defining which classes correspond to which category index
	tr_it, v_it, te_it, c_d = functions.prepare_data(csvpath, frame_size, batch_size)	
	
	#Build model architecture
	INPUT_DIM = frame_size * frame_size * 3
	OUTPUT_DIM = 7
	model = MLP(INPUT_DIM, OUTPUT_DIM)	

	#train model on info in csv
	output_filename = ''+resultspath+"/"+model.name()
	
	#delete any previous model with the same name
	try:
		os.remove(output_filename)
		print("Checkpoint file removed")
	except OSError:
		pass
		
	#training function
	functions.train_model(num_epochs, model, tr_it, v_it, output_filename, trial_num, False)

	#load trained model from pt, test model
	cm = functions.test_model(output_filename, model, te_it)
	
	write_results_to_csv(cm, output_filename, trial_num, c_d, frame_size, num_epochs, batch_size)

if __name__ == "__main__":
    main()
    

