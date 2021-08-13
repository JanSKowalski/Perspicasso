import functions

datapath = "../Dataset/images"
csvpath = "./images.csv"

##########################################################################
##############             Main commands                 #################
##########################################################################
def main():
	#prepare data as csv
	#functions.write_art_labels_to_csv(datapath, csvpath)
		
	#data into dataloaders, load model specs
	#returns model architecture and data iterators
	model, tr_it, v_it, te_it = functions.prepare_model(csvpath)	
		
	#train model on info in csv
	#functions.train_model(model, tr_it, v_it, te_it)

	#autogenerate graphs for analysis
	#matplotlib of train/validation accuracy as epochs go on

##########################################################################


if __name__ == "__main__":
    main()
    

