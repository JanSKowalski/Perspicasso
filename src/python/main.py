import functions

datapath = "../../Dataset/images"
csvpath = "./images.csv"

##########################################################################
##############             Main commands                 #################
##########################################################################
def main():
	#prepare data
	
	#write_csv script
	#functions.write_art_labels_to_csv(datapath, csvpath)
		
	#train model
	functions.train_model(csvpath)

	#autogenerate graphs for analysis
	#matplotlib of train/validation accuracy as epochs go on

##########################################################################


if __name__ == "__main__":
    main()
    

