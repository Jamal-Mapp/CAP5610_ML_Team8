CAP5610_ML_Team8

# For CNN  
This set up requires for the dataset to be pre-donwloaded and unzipped in a directory name asl-signs.  
the dataset can be found at https://www.kaggle.com/competitions/asl-signs/data  
Once the dataset is present and unzipped, simply run "python3 train_asl_model.py"  
The model will beginning training  
The data set is extremely large so please plan accordingly.  


# For Transformers  
install dependencies
pip install -r requirements.txt

setup kaggle api tokens before download the data
Download training data using: kaggle competitions download -c asl-signs
Download testing data from here: kaggle datasets download sohier/461054610546105

# put train data to folder 'train', and test data to folder 'test' and update the ASL_DIR in preprocess.py
python preprocess.py

train
python trainer.py

#The weight can be downloaded from here: https://ucf-my.sharepoint.com/:u:/g/personal/ze123631_ucf_edu/Eek3F5dcy2tPi8pv_WLIt10Bu9pHv_qIzK9RJIuGM-WIgg?e=u2K2G6.  
#link will expire after 12/31/2024  
#test
python test.py

# For LLM  
Download training data using: kaggle competitions download -c asl-signs Download testing data from here: kaggle datasets download sohier/461054610546105  

#Run the file python train.py

# **MLP Model for ASL Sign Classification**

## **Project Description**

This project implements a Multi-Layer Perceptron (MLP) model to classify American Sign Language (ASL) hand signs into 29 categories. The model uses high-dimensional feature vectors as input and outputs the predicted class probabilities for each sample. The MLP serves as a baseline model for evaluating approaches to ASL sign classification.

## **Environment Requirements**

### **1\. Programming Language**

- Python 3.8 or higher.

### **2\. Required Libraries**

To run the code, you need to install the following dependencies:

pip install torch numpy matplotlib scikit-learn joblib

### **3\. Hardware Requirements**

- **Processor**: Intel Core i5 or higher recommended.
- **RAM**: Minimum 8 GB (16 GB recommended for faster processing).
- **GPU**: Optional. If available, use an NVIDIA GPU with CUDA for accelerated training.

## **Data Preparation**

### **1\. Input Files**

Place the following files in the working directory:

- feature_data.npy: Contains the feature vectors of shape (94477,3258).
- feature_labels.npy: Contains the corresponding labels (29 classes).

### **2\. Preprocessing**

The following preprocessing steps are applied in the code:

1. **Standardization**:
    1. Features are normalized using StandardScaler to have zero mean and unit variance.
2. **Label Encoding**:
    1. Labels are converted to integer representations using LabelEncoder.

Preprocessed objects (scaler.pkl and label_encoder.pkl) are saved for reuse.

## **Model Overview**

The MLP model consists of:

1. **Input Layer**: Accepts a feature vector of size _32583258_3258.
2. **Hidden Layers**:
    1. Two hidden layers, each with:
        1. 512 neurons.
        2. ReLU activation.
        3. Batch Normalization.
        4. Dropout (rate = 0.3).
3. **Output Layer**:
    1. 29 neurons with softmax activation to output probabilities for each class.

During training, learning rates from 0.1 to 0.0000001 were tested, with 0.0001 yielding the best results. Dropout rates of 0.1, 0.3, and 0.5 were evaluated, with 0.3 achieving the optimal balance between underfitting and overfitting.

## **How to Run**

### **1\. Train the Model**

Run the following command to train the MLP model:

python train_mlp.py

### **2\. Output Files**

After training, the following files will be generated:

- **Trained Model**: best_model.pth.
- **Scaler and Label Encoder**: scaler.pkl, label_encoder.pkl.
- **Visualizations**: Loss, accuracy, and F1-score curves for training and validation.

### **3\. View Results**

The final results, including training and validation accuracy, loss, and F1-score, are printed in the terminal and visualized using Matplotlib.

## **Model Performance**

- **Training Accuracy**: 69.31%.
- **Validation Accuracy**: 67.85%.
- **Validation F1-Score**: 0.6726.

The model achieves good convergence with minimal overfitting, as indicated by the steady loss and accuracy curves.

The validation F1-Score of 0.6726 suggests the model performs well across classes, though further improvement could address class imbalances.

## **Additional Notes**

- To modify hyperparameters such as learning rate, batch size, or the number of epochs, update the corresponding variables in train_mlp.py.
- Ensure all dependencies are installed and input files are correctly placed in the working directory before running the code.  

## **Model Testing and Evaluation**
After training the MLP model, I attempted to test its performance using a separate test dataset. However, I encountered several challenges during this phase. The main issue was the absence of feature data files; the test data file labels.parquet only contained the columns 'path', 'sign', and 'Usage', but the model required an input feature matrix with 3258 features. Without the actual feature data, the model could not make predictions, causing errors and halting progress.  

I tried to load the feature data from the paths specified in the 'path' column, but these files were either nonexistent or inaccessible, making it impossible to provide the necessary input features for model testing. Additionally, I faced memory management issues when attempting to load all feature data into memory simultaneously. Implementing an on-demand data loading strategy helped reduce memory usage.  

There were also label encoding inconsistencies; the LabelEncoder was fitted on numerical labels during training, but the test data contained string labels like 'go' and 'read'. I addressed this by ensuring that both training and testing used string labels and the same LabelEncoder. However, the test data included unseen labels not present in the training data, causing further encoding issues, which I mitigated by filtering out these unknown labels during testing.  

In summary, while the training process was successful and involved careful hyperparameter tuning to optimize the model's performance, the testing phase highlighted the importance of having complete and consistent datasets. For future projects, I plan to secure all necessary data files, optimize memory management when handling large-scale data, ensure data consistency across different stages, and incorporate thorough data validation steps to promptly identify and resolve issues.  
