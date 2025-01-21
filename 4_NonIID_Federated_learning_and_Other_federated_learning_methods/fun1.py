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

def df_xy(pd_train1, col_num):
    encoder = LabelEncoder()
    pd_train1['GENDER'] = encoder.fit_transform(pd_train1['GENDER'])
    pd_train1_cols = pd_train1.columns.tolist()
    pd_train_x1_cols = pd_train1_cols[col_num:]
    
    pd_train_x = pd_train1[pd_train_x1_cols]
    pd_train_x[pd_train_x1_cols] = MinMaxScaler().fit_transform(
        pd_train_x[pd_train_x1_cols])
    pd_train_x2 = pd_train_x.values
    
    pd_train_y = pd_train1['aki_label'].values
    lb = LabelBinarizer()
    pd_train_y2 = lb.fit_transform(pd_train_y)
    return pd_train_x2, pd_train_y2


def accuracy(model_types, model5, pd_test_xi, pd_test_yi):
    
    if model_types == 'nn' or model_types == 'rnn':
        preds3i = model5.predict(pd_test_xi)
    if model_types == 'xgb':
        preds3i = model5.predict(xgb.DMatrix(pd_test_xi))
         
    ts_preds = []
    for i in preds3i:
        if i > 0.5:
            preds = 1
        else:
            preds = 0
        ts_preds.append(preds) 
    ts_aki1_Acc = accuracy_score(ts_preds, pd_test_yi)
    ts_aki2_Acc = round(ts_aki1_Acc*100, 2)
    return ts_aki2_Acc


def roc_auc_and_pr_auc(model_types, model5, 
                       pd_test_xi, pd_test_yi):
    
    if model_types == 'nn':
        preds3i = model5.predict(pd_test_xi)
        
    if model_types == 'xgb':
        preds3i = model5.predict(xgb.DMatrix(pd_test_xi))
        
    if model_types == 'rnn':
        preds2i = model5.predict(pd_test_xi)
        preds3i = []
        for i in preds2i:
            preds3i.append(i[0])        
        
    fpr, tpr, thresholds = roc_curve(pd_test_yi, preds3i) 
    roc_auc = auc(fpr, tpr)
    roc_auc2 = round(roc_auc*100, 2)
    pre, re, thresholds2 = precision_recall_curve(pd_test_yi, preds3i) 
    pr_auc = auc(re, pre)
    pr_auc2 = round(pr_auc*100, 2)
    return roc_auc2, pr_auc2


def plot_training_process(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b-', label='Training acc')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.legend()
    plt.show()

    
def get_upsampled_df(aki_df):
    aki0_df1 = aki_df[aki_df['aki_label'] == 0]
    aki1_df1 = aki_df[aki_df['aki_label'] == 1]
    aki0_shape = aki0_df1.shape[0]
    aki1_shape = aki1_df1.shape[0]

    if aki0_shape > aki1_shape:
        sample_num = aki0_shape - aki1_shape
        aki1_df2 = aki1_df1.sample(
            n = sample_num, replace=True, random_state=42)
        aki_df2 = pd.concat([aki1_df1, aki1_df2, aki0_df1])
        aki_df3 = aki_df2.sample(
            frac=1, random_state=42).reset_index(drop=True)

    if aki1_shape > aki0_shape:
        sample_num = aki1_shape - aki0_shape
        aki0_df2 = aki0_df1.sample(
            n = sample_num, replace=True, random_state=42)
        aki_df2 = pd.concat([aki0_df1, aki0_df2, aki1_df1]) 
        aki_df3 = aki_df2.sample(
            frac=1, random_state=42).reset_index(drop=True)
        
    return aki_df2

def get_downsampled_df(aki_df):
    aki0_df1 = aki_df[aki_df['aki_label'] == 0]
    aki1_df1 = aki_df[aki_df['aki_label'] == 1]
    aki0_shape = aki0_df1.shape[0]
    aki1_shape = aki1_df1.shape[0]

    if aki0_shape > aki1_shape:
        aki0_df2 = aki0_df1.sample(
            n = aki1_shape, replace=True, random_state=42)
        aki_df2 = pd.concat([aki0_df2, aki1_df1])

    if aki1_shape > aki0_shape:
        aki1_df2 = aki1_df1.sample(
            n = aki0_shape, replace=True, random_state=42)
        aki_df2 = pd.concat([aki0_df1, aki1_df2])
        
    return aki_df2


def get_aki_num(aki_df):
    aki0_df1 = aki_df[aki_df['aki_label'] == 0]
    aki1_df1 = aki_df[aki_df['aki_label'] == 1]
    aki0_shape = aki0_df1.shape[0]
    aki1_shape = aki1_df1.shape[0]
    return aki0_shape, aki1_shape

def get_aki_01_row(aki_df):
    aki0_df1 = aki_df[aki_df['aki_label'] == 0]
    aki1_df1 = aki_df[aki_df['aki_label'] == 1]
    return aki0_df1, aki1_df1


