This set up requires for the dataset to be pre-donwloaded and unzipped in a directory name asl-signs.
the dataset can be found at https://www.kaggle.com/competitions/asl-signs/data
The testing dataset can be found at https://www.kaggle.com/datasets/sohier/461054610546105
When unzipping the training dataset, everything should already be inside the asl-signs folder
Create a folder called "test-ml" and unzip all the contents of the testing dataset and put it in there.
Once the dataset is present and unzipped, make sure the tf-lstm-final.ipynb is in the same level as asl-signs and testing data.

create "asl-signs" and "test-ml" if they are not already there where "asl-signs" will contain the training data and "test-ml" which will contain the testing data respectively.  
asl-signs should contain train.csv (which has the labels and formatting), 
sign\_to\_prediction\_index\_map.json (containing the 250 classes), 
and train\_landmark\_files (the actual dataset in a parquet format). 
the test-ml should contain 3 things, labels.parquet (which has the labels and formatting), 
sign\_to\_prediction\_index\_map.json (containing the 250 classes), 
and test\_landmark\_files (the actual dataset in a parquet format).

For the training and testing landmark folders, they contain many folders which are numbered which have parquet files inside.
The data is automatically pulled from there and in the python file
pip install the libraries above and install python 3.9.2. then you can run the file


This Project utilizes Python 3.9.2 alongside Tensorflow which is a library for building and training the model. 
I will be utilizing tensorflow-keras 2.10.0 on my computer with a GPU as it was easiest to work on this slightly older version of python as 
tensorflow-keras 2.10.0 was the last version to be updated on windows to have GPU support built into the library. 
Due to this, I had to downgrade NumPy for numerical computations, Matplotlib for plotting learning curves, and Pandas for data manipulation and cleaning 
to work with this tensorflow and python version. I used NumPy 1.22.4, Pandas 1.3.5, and matplotlib 3.8.0 in this project. 
Finally, I installed scipy and scikit learn for access to more robust statistics and data preprocessing and will be using scipy 1.13.1 and scikit-learn 1.5.2.
Finally, these were all installed via a python virtualenv which can be setup in anaconda or command line to manage all dependencies and not conflict with other projects.

\begin{verbatim}
!pip install numpy==1.22.4
!pip install pandas==1.3.5
!pip show tensorflow #(Name: tensorflow, Version: 2.10.0)
!pip install tqdm
!pip install scikit-learn==1.5.2
!pip install scipy==1.13.1
!pip install tslearn plotly #additional visualization
!pip install pyarrow
!pip install matplotlib==3.8.0
import sys print(sys.version) #3.9.20 
\end{verbatim}

 
Define and train the LSTM model then save the best model and preprocessing tools. 
Optionally, if a GPU is available, leverage it to accelerate training the model 
(one should be able to do so as long as they has access to a device/network with a GPU). 
Ensure the data is properly cleaned and preprocessed for optimal performance, and use a validation set to monitor and prevent overfitting. 
