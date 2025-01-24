# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def df_xy(pd_train1):
    """
    Preprocesses the training dataframe by encoding categorical variables and scaling numerical features.
    
    Args:
        pd_train1 (pd.DataFrame): Input training dataframe with features and target variable
        
    Returns:
        tuple: (X, y) where X is the preprocessed feature matrix and y is the binary encoded target variable
    """
    # Encode gender column
    encoder = LabelEncoder()
    pd_train1['GENDER'] = encoder.fit_transform(pd_train1['GENDER'])
    
    # Select feature columns (assumes first 5 columns are non-feature columns)
    pd_train1_cols = pd_train1.columns.tolist()
    pd_train_x1_cols = pd_train1_cols[5:]
    
    # Scale features to [0,1] range
    pd_train_x = pd_train1[pd_train_x1_cols]
    pd_train_x[pd_train_x1_cols] = MinMaxScaler().fit_transform(
        pd_train_x[pd_train_x1_cols])
    pd_train_x2 = pd_train_x.values
    
    # Convert target variable to binary format
    pd_train_y = pd_train1['aki_label'].values
    lb = LabelBinarizer()
    pd_train_y2 = lb.fit_transform(pd_train_y)
    return pd_train_x2, pd_train_y2

def accuracy(model_types, model5, pd_test_xi, pd_test_yi):
    """
    Calculates the accuracy score for different types of models.
    
    Args:
        model_types (str): Type of model ('nn', 'rnn', or 'xgb')
        model5: Trained model object
        pd_test_xi: Test features
        pd_test_yi: True test labels
        
    Returns:
        float: Accuracy score as a percentage (0-100)
    """
    # Get predictions based on model type
    if model_types == 'nn' or model_types == 'rnn':
        preds3i = model5.predict(pd_test_xi)
    if model_types == 'xgb':
        preds3i = model5.predict(xgb.DMatrix(pd_test_xi))
    
    # Convert probabilities to binary predictions using 0.5 threshold
    ts_preds = []
    for i in preds3i:
        if i > 0.5:
            preds = 1
        else:
            preds = 0
        ts_preds.append(preds) 
    
    # Calculate and round accuracy score
    ts_aki1_Acc = accuracy_score(ts_preds, pd_test_yi)
    ts_aki2_Acc = round(ts_aki1_Acc*100, 2)
    return ts_aki2_Acc

def roc_auc_and_pr_auc(model_types, model5, pd_test_xi, pd_test_yi):
    """
    Calculates ROC-AUC and PR-AUC scores for model evaluation.
    
    Args:
        model_types (str): Type of model ('nn', 'rnn', or 'xgb')
        model5: Trained model object
        pd_test_xi: Test features
        pd_test_yi: True test labels
        
    Returns:
        tuple: (ROC-AUC score, PR-AUC score) as percentages (0-100)
    """
    # Get predictions based on model type
    if model_types == 'nn':
        preds3i = model5.predict(pd_test_xi)
    elif model_types == 'xgb':
        preds3i = model5.predict(xgb.DMatrix(pd_test_xi))
    elif model_types == 'rnn':
        preds2i = model5.predict(pd_test_xi)
        preds3i = []
        for i in preds2i:
            preds3i.append(i[0])        
    
    # Calculate ROC-AUC
    fpr, tpr, thresholds = roc_curve(pd_test_yi, preds3i) 
    roc_auc = auc(fpr, tpr)
    roc_auc2 = round(roc_auc*100, 2)
    
    # Calculate PR-AUC
    pre, re, thresholds2 = precision_recall_curve(pd_test_yi, preds3i) 
    pr_auc = auc(re, pre)
    pr_auc2 = round(pr_auc*100, 2)
    return roc_auc2, pr_auc2

def plot_training_process(history):
    """
    Plots the training accuracy and loss curves from model training history.
    
    Args:
        history: Training history object from model.fit()
    """
    # Plot accuracy
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b-', label='Training acc')
    plt.legend()
    plt.figure()
    
    # Plot loss
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.legend()
    plt.show()

def get_upsampled_df(aki_df):
    """
    Performs upsampling on the minority class to balance the dataset.
    
    Args:
        aki_df (pd.DataFrame): Input dataframe with 'aki_label' column
        
    Returns:
        pd.DataFrame: Balanced dataframe with upsampled minority class
    """
    # Split data by class
    aki0_df1 = aki_df[aki_df['aki_label'] == 0]
    aki1_df1 = aki_df[aki_df['aki_label'] == 1]
    aki0_shape = aki0_df1.shape[0]
    aki1_shape = aki1_df1.shape[0]

    # Upsample minority class to match majority class size
    if aki0_shape > aki1_shape:
        sample_num = aki0_shape - aki1_shape
        aki1_df2 = aki1_df1.sample(n=sample_num, replace=True, random_state=42)
        aki_df2 = pd.concat([aki1_df1, aki1_df2, aki0_df1])
        aki_df3 = aki_df2.sample(frac=1, random_state=42).reset_index(drop=True)
    elif aki1_shape > aki0_shape:
        sample_num = aki1_shape - aki0_shape
        aki0_df2 = aki0_df1.sample(n=sample_num, replace=True, random_state=42)
        aki_df2 = pd.concat([aki0_df1, aki0_df2, aki1_df1]) 
        aki_df3 = aki_df2.sample(frac=1, random_state=42).reset_index(drop=True)
        
    return aki_df2

def get_downsampled_df(aki_df):
    """
    Performs downsampling on the majority class to balance the dataset.
    
    Args:
        aki_df (pd.DataFrame): Input dataframe with 'aki_label' column
        
    Returns:
        pd.DataFrame: Balanced dataframe with downsampled majority class
    """
    # Split data by class
    aki0_df1 = aki_df[aki_df['aki_label'] == 0]
    aki1_df1 = aki_df[aki_df['aki_label'] == 1]
    aki0_shape = aki0_df1.shape[0]
    aki1_shape = aki1_df1.shape[0]

    # Downsample majority class to match minority class size
    if aki0_shape > aki1_shape:
        aki0_df2 = aki0_df1.sample(n=aki1_shape, replace=True, random_state=42)
        aki_df2 = pd.concat([aki0_df2, aki1_df1])
    elif aki1_shape > aki0_shape:
        aki1_df2 = aki1_df1.sample(n=aki0_shape, replace=True, random_state=42)
        aki_df2 = pd.concat([aki0_df1, aki1_df2])
        
    return aki_df2

def get_aki_num(aki_df):
    """
    Gets the count of samples in each class.
    
    Args:
        aki_df (pd.DataFrame): Input dataframe with 'aki_label' column
        
    Returns:
        tuple: (count of class 0 samples, count of class 1 samples)
    """
    aki0_df1 = aki_df[aki_df['aki_label'] == 0]
    aki1_df1 = aki_df[aki_df['aki_label'] == 1]
    aki0_shape = aki0_df1.shape[0]
    aki1_shape = aki1_df1.shape[0]
    return aki0_shape, aki1_shape

def get_aki_01_row(aki_df):
    """
    Splits the dataframe into separate dataframes for each class.
    
    Args:
        aki_df (pd.DataFrame): Input dataframe with 'aki_label' column
        
    Returns:
        tuple: (dataframe with class 0 samples, dataframe with class 1 samples)
    """
    aki0_df1 = aki_df[aki_df['aki_label'] == 0]
    aki1_df1 = aki_df[aki_df['aki_label'] == 1]
    return aki0_df1, aki1_df1

