import functions
import csv
import os
import itertools
from classes import MLP
from functions import continue_training_model
import subprocess


datapath = "../Dataset/images"
csvpath = "./images.csv"
outputpath = "./plotting.csv"

##########################################################################
##############             Main commands                 #################
##########################################################################
def plotting_script():

	#prepare data as csv
	functions.write_art_labels_to_csv(datapath, csvpath)
		

	
	csv_file = open(outputpath, 'w')
	writer = csv.writer(csv_file)
	
	frame_size_range = range(250,260,130)
	num_epochs_step = 10
	num_epochs_range = range(5,20,num_epochs_step)
	for frame_size, num_epochs in itertools.product(frame_size_range, num_epochs_range):
		print(f"---------- Frame size: {frame_size}, Num epochs: {num_epochs} -----------")
		
		#Build model architecture
		INPUT_DIM = frame_size * frame_size * 3
		OUTPUT_DIM = 7
		model = MLP(INPUT_DIM, OUTPUT_DIM)	
		
		if (num_epochs == 5):
			#split train/val/test, then load into data iterators
			tr_it, v_it, te_it = functions.prepare_data(csvpath, frame_size)
		

			#train model on info in csv
			output_filename = model.name()+'.pt'
			functions.train_model(num_epochs, model, tr_it, v_it, output_filename)
		else:
			output_filename = model.name()+'.pt'
			continue_training_model(num_epochs_step, model, tr_it, v_it, output_filename)
			
		accuracy = functions.test_model(output_filename, model, te_it)
		row = [num_epochs, frame_size, accuracy]
		writer.writerow(row)
	
	csv_file.close()
	subprocess.run(["gnuplot", "Image_accuracy_vs_epoch.plot"])
	subprocess.run(["xdg-open", "test.png"])
	
	'''
	
	while (frame_size <= 40):
		print(f"-----------New frame size: %d -----------" % frame_size)
		num_epochs = 10
		#split train/val/test, then load into data iterators
		tr_it, v_it, te_it = functions.prepare_data(csvpath, frame_size)
		
		#Build model architecture
		INPUT_DIM = frame_size * frame_size * 3
		OUTPUT_DIM = 7
		model = MLP(INPUT_DIM, OUTPUT_DIM)	
			
		while (num_epochs <= 20):

			#train model on info in csv
			output_filename = model.name()+'.pt'
			functions.train_model(num_epochs, model, tr_it, v_it, output_filename)

			#load trained model from pt, test model
			accuracy = functions.test_model(output_filename, model, te_it)

			row = [num_epochs, frame_size, accuracy]
			writer.writerow(row)
			num_epochs += 10
		frame_size += 80
	csv_file.close()


	os.system("gnuplot Accuracy_vs_epoch_vs_frame_size.plot")
	
	'''

##########################################################################


if __name__ == "__main__":
    plotting_script()
    

