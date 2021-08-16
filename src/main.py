import functions
from classes import MLP

datapath = "../Dataset/images"
csvpath = "./images.csv"

##########################################################################
##############             Main commands                 #################
##########################################################################
def main():

	#prepare data as csv
	functions.write_art_labels_to_csv(datapath, csvpath)
	
	print(functions.largest_frame_size(csvpath))
		
	#split train/val/test, then load into data iterators
	#tr_it, v_it, te_it = functions.prepare_data(csvpath)	
	
	#Build model architecture
	INPUT_DIM = 28 * 28 * 3
	OUTPUT_DIM = 7
	#model = MLP(INPUT_DIM, OUTPUT_DIM)	

	#train model on info in csv
	NUM_EPOCHS = 600
	#output_filename = model.name()+'.pt'
	#functions.train_model(NUM_EPOCHS, model, tr_it, v_it, output_filename)

	#load trained model from pt, test model
	#functions.test_model(output_filename, model, te_it)



##########################################################################


if __name__ == "__main__":
    main()
    

