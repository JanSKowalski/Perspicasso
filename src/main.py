#! /home/jan/anaconda3/bin/python3

import functions
import csv
import os
import itertools
import functions
from classes import MLP, MLP_w_torchlrp
import subprocess

import os
import sys
import torch
import random
import pathlib
import argparse
import torchvision
import matplotlib.pyplot as plt
from utils import store_patterns, load_patterns
from visualization import heatmap_grid
import lrp
from lrp.patterns import fit_patternnet, fit_patternnet_positive # PatternNet patterns

import torch.nn as nn

datapath = "../Dataset/images"
csvpath = "./images.csv"
outputpath = "./plotting.csv"

##########################################################################
##############             Main commands                 #################
##########################################################################
def main():
	simple_example()
	#lrp_example()
	#repeated_iterations()
	

##########################################################################

#fixed frame size, fixed number of epochs
def simple_example():
	#prepare data as csv
	functions.write_art_labels_to_csv(datapath, csvpath)
	
	#set values for frame size and num epochs
	frame_size = 160
	num_epochs = 2
		
	#split train/val/test, then load into data iterators
	tr_it, v_it, te_it = functions.prepare_data(csvpath, frame_size)	
	
	#Build model architecture
	INPUT_DIM = frame_size * frame_size * 3
	OUTPUT_DIM = 7
	model = MLP_w_torchlrp(INPUT_DIM, OUTPUT_DIM)	

	#train model on info in csv
	output_filename = model.name()+'.pt'
	
	#delete any previous model with the same name
	try:
		os.remove(output_filename)
		print("Checkpoint file removed")
	except OSError:
		pass
		
	#training function
	functions.train_model(num_epochs, model, tr_it, v_it, output_filename)

	#load trained model from pt, test model
	functions.test_model(output_filename, model, te_it)

#fixed frame size, fixed number of epochs
def lrp_example():
	#prepare data as csv
	functions.write_art_labels_to_csv(datapath, csvpath)
	
	#set values for frame size and num epochs
	frame_size = 160
	num_epochs = 2
		
	#split train/val/test, then load into data iterators
	tr_it, v_it, te_it = functions.prepare_data(csvpath, frame_size)	
	
	#Build model architecture
	INPUT_DIM = frame_size * frame_size * 3
	OUTPUT_DIM = 7
	model = nn.Sequential(
		lrp.Linear(INPUT_DIM, 250),
		torch.nn.ReLU(),
		lrp.Linear(250, 100),
		torch.nn.ReLU(),
		lrp.Linear(100, OUTPUT_DIM)		
	)
	#model = MLP_w_torchlrp(INPUT_DIM, OUTPUT_DIM)	

	#train model on info in csv
	output_filename = model.name()+'.pt'
	
	#delete any previous model with the same name
	try:
		os.remove(output_filename)
		print("Checkpoint file removed")
	except OSError:
		pass
		
	#training function
	functions.train_model(num_epochs, model, tr_it, v_it, output_filename)

	#load trained model from pt, test model
	functions.test_model(output_filename, model, te_it)
	
	with torch.no_grad(): 
		y_hat = model(x)
		pred = y_hat.max(1)[1]

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


	# # # # Patterns for PatternNet and PatternAttribution
	all_patterns_path = (base_path / 'examples' / 'patterns' / 'pattern_all.pkl').as_posix()
	if not os.path.exists(all_patterns_path):  # Either load of compute them
		patterns_all = fit_patternnet(model, train_loader, device=args.device)
		store_patterns(all_patterns_path, patterns_all)
	else:
		patterns_all = [torch.tensor(p, device=args.device, dtype=torch.float32) for p in load_patterns(all_patterns_path)]

	pos_patterns_path = (base_path / 'examples' / 'patterns' / 'pattern_pos.pkl').as_posix()
	if not os.path.exists(pos_patterns_path):
		patterns_pos = fit_patternnet_positive(model, train_loader, device=args.device)#, max_iter=1)
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

#allow both frame size and num epochs to vary
def repeated_iterations():
	#prepare data as csv
	functions.write_art_labels_to_csv(datapath, csvpath)
		
	csv_file = open(outputpath, 'w')
	writer = csv.writer(csv_file)
	
	#frame_size_range = range(40,330,40)
	#first_epoch_value = 20
	#num_epochs_step = 10
	#num_epochs_range = range(first_epoch_value, 610,num_epochs_step)
	frame_size_range = range(160,165,40)
	first_epoch_value = 20
	num_epochs_step = 10
	num_epochs_range = range(first_epoch_value, 30,num_epochs_step)
	for frame_size, num_epochs in itertools.product(frame_size_range, num_epochs_range):
		print(f"---------- Frame size: {frame_size}, Num epochs: {num_epochs} -----------")
		
		#Build model architecture
		INPUT_DIM = frame_size * frame_size * 3
		OUTPUT_DIM = 7
		model = MLP(INPUT_DIM, OUTPUT_DIM)	
				
		output_filename = model.name()+'.pt'

		#prepare data iterators if this is the first 
		if (num_epochs == first_epoch_value):
			#split train/val/test, then load into data iterators
			tr_it, v_it, te_it = functions.prepare_data(csvpath, frame_size)
			
			#delete any previous model with the same name
			try:
				os.remove(output_filename)
				print("Checkpoint file removed")
			except OSError:
				pass

		#train model on info in csv
		functions.train_model(num_epochs, model, tr_it, v_it, output_filename)
			
		accuracy = functions.test_model(output_filename, model, te_it)
		row = [num_epochs, frame_size, accuracy]
		writer.writerow(row)
	
	csv_file.close()
	#subprocess.run(["gnuplot", "Image_accuracy_vs_epoch.plot"])
	#subprocess.run(["xdg-open", "test.png"])

if __name__ == "__main__":
    main()
    

