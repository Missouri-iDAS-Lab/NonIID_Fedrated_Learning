# Non-IID Degree Estimation and Non-IID Federated Learning 

This repository contains the Jupyter Notebooks of the paper "Effective Non-IID Degree Estimation for Robust Federated Learning in Healthcare Datasets"

## Table of Contents

Introduction

Repository Overview

Installation and Requirements

Data Availability and Access

Key Methods and Algorithms

Experimental Design

Results and Evaluation

Usage Guide

Citation

## 1. Introduction

Non-independent and non-identically distributed (non-IID) data often arises in healthcare due to variations in patient demographics, clinical practices, environmental factors, and more. This distribution shift can hamper the performance and fairness of machine learning (ML) models.

In federated learning (FL) settings, data resides in multiple healthcare sites (or nodes), each with potentially unique, heterogeneous distributions. While FL allows collaborative model training without directly sharing sensitive data, it does not inherently quantify or mitigate the degree of distribution shift among sites.

### What We Propose
#### 1. Non-IID Degree Estimation

- A statistical method using hypothesis testing and effect size to quantify distribution shifts among local datasets.
- Interpretable, model-agnostic approach for mixed data types.
- Provides stable, normalized estimates of data heterogeneity.
  
#### 2. Evaluation Metrics for Non-IID Estimation

- Variability: Measures consistency of non-IID degree estimates under different sample sizes.
- Separability: Assesses how well a non-IID estimation method distinguishes within-dataset vs. between-dataset differences.
- Computational Time: Tracks efficiency of computing the non-IID degree.

#### 3. Non-IID Federated Learning (Non-IID FL)

- Incorporates the non-IID degree as a regularization term to weigh local updates.
- Limits impact from nodes with large distribution differences, improving robustness and fairness.
  
## 2. Repository Overview
This repository is organized into four main folders reflecting the pipeline of our experiments:

### 2.1. 1_Data_preprocessing_with_Time_sequence_processing_and_Imputation

- Jupyter Notebooks and Python scripts for data cleaning and feature preprocessing.
- Includes time-sequence processing, missing data imputation (e.g., MICE), and outlier removal.
- Final outputs are normalized EHR features for AKI risk prediction.
  
### 2.2. 2_Comparison_of_different_nonIID_estimation_methods

- Implements the proposed statistical-based non-IID estimation method, as well as previous non-IID measurement approaches.
- Contains scripts to generate numerical comparisons of variability, separability, and computational costs.
  
### 2.3. 3_Relationship_between_testing_error_and_nonIID

- Demonstrates the positive correlation between non-IID degree and model testing error.
- Includes evaluations using different ML models (e.g., neural networks, SVM, random forests, XGBoost) to observe performance under distribution shifts.
  
### 2.4. 4_NonIID_Federated_learning_and_Other_federated_learning_methods

- Code for federated learning experiments, including FedAvg, FedProx, Mime Lite, and our proposed Non-IID FL.
- Scripts to compare local, centralized, and federated models on MIMIC-III, MIMIC-IV, and eICU-CRD datasets.
- Demonstrates how incorporating the non-IID degree as a regularization term improves global model performance.
  
## 3. Installation and Requirements
- Python 3.7+
- TensorFlow (tested on version 2.x)
- TensorFlow Federated (for FL simulations, tested on version 0.x)
- NumPy, Pandas, Matplotlib, scikit-learn
- MICE or relevant library for data imputation (e.g., FancyImpute or a built-in function if available)
  
You can install the core dependencies via:
`pip install -r requirements.txt`

## 4. Data Availability and Access
We use three publicly available intensive care unit (ICU) databases hosted on PhysioNet:

### 4.1. MIMIC-III (v1.4)

- https://physionet.org/content/mimiciii/1.4/
- Data collected from 2001 to 2012 at the Beth Israel Deaconess Medical Center.
  
### 4.2. MIMIC-IV (v3.1)

- https://physionet.org/content/mimiciv/3.1/
- Data collected from 2008 to 2019 at the same hospital.
  
### 4.3. eICU Collaborative Research Database (eICU-CRD) (v2.0)

- https://physionet.org/content/eicu-crd/2.0/
- Multi-center ICU data from 2014 to 2015 across 208 hospitals in the US.
  
### Access Requirements
All of these datasets contain sensitive patient information and require credentialed access through PhysioNet. Researchers must complete CITI training and agree to PhysioNet’s Data Use Agreement.

Note: This repository does not provide the original EHR data due to privacy and regulatory constraints. Users must independently obtain these datasets.

## 5. Key Methods and Algorithms
### 5.1. Non-IID Degree Estimation
#### 5.1.1. Statistical Hypothesis Testing:

- Uses tests (e.g., t-tests, chi-square for categorical) to detect distribution differences between local datasets.
- Evaluates the significance of feature-wise distribution shifts.
  
#### 5.1.2. Effect Size Calculation:

- Measures the magnitude of difference (e.g., Cohen’s d, Cramér's V) for numerical or categorical features.
- Aggregates feature-wise effect sizes into a composite non-IID metric.
  
#### 5.1.3. Interpretability and Normalization:

- Final non-IID degree values are normalized to [0, 1].
- Feature-level contributions highlight which variables contribute the most to distribution differences.
  
### 5.2. Metrics for Non-IID Method Evaluation

- Variability: Consistency of non-IID degree estimates under repeated random sampling.
- Separability: Ability to distinguish within-dataset vs. between-dataset differences.
- Computational Time: Efficiency in computing non-IID degree.
  
### 5.3. Non-IID Federated Learning (Non-IID FL)

- Extends FedAvg by including a non-IID regularization term.
- Nodes with higher non-IID degree have a larger regularization factor to reduce their influence on the global model.
- Alleviates the adverse effects of outlier or highly divergent local updates.
  
## 6. Experimental Design

Experiments are designed to assess both non-IID degree estimation and federated learning performance in predicting acute kidney injury (AKI) using MIMIC-III, MIMIC-IV, and eICU-CRD. Key steps:

### 6.1. Data Preprocessing

- Select statistically significant features via Pearson correlation.
- Split records into 6-hour time windows.
- Impute missing data with MICE.
- Remove outliers outside 1st and 99th percentiles.
- Normalize numerical features to [0, 1].
- Label AKI onset for next 6, 12, 24, 48 hours based on KDIGO criteria.
  
### 6.2. Training/Testing Split

- 80% for training, 20% for testing.
- 5-fold cross-validation for model evaluation.
  
### 6.3. Model Architectures

- Local/cross-site training: Simple 3-layer neural network.
- FL: FedAvg, Weighted FedAvg, FedProx, Mime Lite, and Non-IID FL.
- Additional tests with CNN, RNN, LSTM, Decision Tree, Random Forest, XGBoost, etc. (for relationship analysis between non-IID degree and test error).
  
### 6.4.Comparison Settings

- Local Learning: Train only on each local dataset.
- Centralized Learning: Combine all local datasets into one global dataset.
- Federated Learning: Train FL models using local site updates without sharing raw data.
  
## 7. Results and Evaluation

### 7.1. Non-IID Degree Estimation

- Our method outperforms previous approaches (He et al., Li et al., Zhao et al.) on variability, separability, and computation time.
- Produces stable, normalized non-IID estimates across different data sampling percentages.
  
### 7.2. Correlation with Testing Error

- Higher non-IID degree between training and testing sets correlates with higher model error rates.
- Observed across multiple ML models (NN, CNN, RNN, LSTM, XGBoost, SVM, Decision Tree, Random Forest).
  
### 7.3. Non-IID Federated Learning

- Incorporating the non-IID degree as a regularization term outperforms standard FedAvg, Weighted FedAvg, FedProx, Mime Lite, and even centralized training under high data heterogeneity.
- Achieves higher test accuracy on AKI prediction tasks across MIMIC-III, MIMIC-IV, and eICU-CRD.
  
## 8. Usage Guide
### 8.1. Cloning the Repository

`git clone https://github.com/Missouri-iDAS-Lab/NonIID_Fedrated_Learning.git

cd NonIID_Fedrated_Learning`

### 8.2. Running the Experiments
#### 8.2.1. Data Preprocessing

- Navigate to "1_Data_preprocessing_with_Time_sequence_processing_and_Imputation/"
- Update paths to point to your local copies of MIMIC-III, MIMIC-IV, eICU-CRD.
- Run the Jupyter Notebooks in order to generate the cleaned, imputed, and normalized data files.
  
#### 8.2.2. Non-IID Degree Comparison

- Go to "2_Comparison_of_different_nonIID_estimation_methods/"
- Execute the notebook(s) to reproduce the comparisons among different non-IID degree estimation algorithms.
- Adjust parameters (e.g., sampling percentages, random seeds) as needed.
  
#### 8.2.3. Relationship Analysis

- In "3_Relationship_between_testing_error_and_nonIID/", run the experiments to see how non-IID degree correlates with test error across multiple ML models.
  
#### 8.2.4. Federated Learning

- In 4_NonIID_Federated_learning_and_Other_federated_learning_methods/, explore scripts for FedAvg, FedProx, Mime Lite, Weighted FL, and our proposed Non-IID FL approach.
- Modify hyperparameters (e.g., learning rate, batch size) in the configuration files or at the top of each notebook.
- Compare final accuracy, AUC, or other metrics for each method.

### 8.3. Customization

- Models: You can easily switch neural networks (CNN, RNN, etc.) by editing the relevant sections in the Jupyter Notebooks.
- Metrics: For advanced statistical tests or effect size calculations, modify the relevant scripts in the 2_Comparison_of_different_nonIID_estimation_methods folder.
- Regularization: The non-IID regularization factor can be tuned in the FL scripts to control how strongly heterogeneous nodes are penalized during global aggregation.
  
## 9. Citation
If you find our work or code helpful for your research, please cite:

"""
@article{chen2023nonIID,
  title={Effective Non-IID Degree Estimation for Robust Federated Learning in Healthcare Datasets},
  author={Chen, Kun-Yi and Shyu, Chi-Ren and Tsai, Yuan-Yu and Baskett, William I. and Chang, Chi-Yu and Chou, Che-Yi and Tsai, Jeffrey J. P. and Shae, Zon-Yin},
  journal={(Please update with final journal info when available)},
  year={2023},
  note={arXiv / preprint / etc. if applicable}
}
"""

### Disclaimer:

- This repository is provided for academic and research purposes.
- Access to MIMIC-III, MIMIC-IV, and eICU-CRD databases requires approval from PhysioNet and completion of CITI training.
- For any questions or suggestions, please feel free to open an issue or contact the authors.
