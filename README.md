CAP5610_ML_Team8

For CNN  
This set up requires for the dataset to be pre-donwloaded and unzipped in a directory name asl-signs.  
the dataset can be found at https://www.kaggle.com/competitions/asl-signs/data  
Once the dataset is present and unzipped, simply run "python3 train_asl_model.py"  
The model will beginning training  
The data set is extremely large so please plan accordingly.  



For Transformers  
install dependencies
pip install -r requirements.txt

setup kaggle api tokens before download the data
Download training data using: kaggle competitions download -c asl-signs
Download testing data from here: kaggle datasets download sohier/461054610546105

train
python trainer.py

#The weight can be downloaded from here: https://ucf-my.sharepoint.com/:u:/g/personal/ze123631_ucf_edu/Eek3F5dcy2tPi8pv_WLIt10Bu9pHv_qIzK9RJIuGM-WIgg?e=u2K2G6.  
#link will expire after 12/31/2024  
#test
python test.py
