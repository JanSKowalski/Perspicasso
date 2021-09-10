#! /home/jan/anaconda3/bin/python3

import functions
import csv
import os
import itertools
import functions
from classes import MLP
import subprocess

datapath = "../Dataset/images"
csvpath = "./images.csv"
outputpath = "./plotting.csv"

##########################################################################
##############             Main commands                 #################
##########################################################################
def main():
	simple_example()

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
	model = MLP(INPUT_DIM, OUTPUT_DIM)	

	#train model on info in csv
	output_filename = model.name()+'.pt'
	functions.train_model(num_epochs, model, tr_it, v_it, output_filename)

	#load trained model from pt, test model
	functions.test_model(output_filename, model, te_it)

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
    

