# Multi-Class and Multi-Label Classification Using SVM and K-Means Clustering

## Project Overview

This project involves **multi-class** and **multi-label classification** using Support Vector Machines (SVM) and **clustering** using K-Means on the **Anuran Calls (MFCCs) Data Set**. The main objective is to solve the multi-label classification problem by training SVMs for each label and evaluating their performance using different metrics. We also explore the use of SMOTE for handling class imbalance and implement K-Means clustering for unsupervised learning.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Task 1: Multi-Class and Multi-Label Classification with SVMs](#task-1-multi-class-and-multi-label-classification-with-svms)
- [Task 2: K-Means Clustering on Multi-Class and Multi-Label Data](#task-2-k-means-clustering-on-multi-class-and-multi-label-data)
- [How to Run](#how-to-run)
- [Results](#results)
- [License](#license)

## Dataset

### Anuran Calls (MFCCs) Dataset
- **Source**: [UCI Machine Learning Repository - Anuran Calls (MFCCs)](https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29)
- **Data Description**: 
  - The dataset contains acoustic features extracted from anuran (frogs) calls.
  - Each instance is associated with three labels: **Family**, **Genus**, and **Species**.
  - The dataset has 7,195 instances and 22 attributes representing **Mel-frequency cepstral coefficients (MFCCs)**.
  
More detailed information about the dataset can be found in the accompanying [Readme file](Readme.txt).

## Task 1: Multi-Class and Multi-Label Classification with SVMs

### (a) Data Preparation
- **Training and Test Split**: 70% of the dataset was randomly selected as the training set.
  
### (b) Binary Relevance Approach
- **Exact Match** and **Hamming Score/Loss** were researched and used to evaluate the performance of multi-label classification.
- **SVMs with Gaussian Kernels** were trained for each label (Family, Genus, and Species) using a **one-versus-all** strategy.
  
### (c) L1-Penalized SVMs
- The task was repeated with **L1-penalized SVMs** to improve performance. Standardized attributes were used for training.
  
### (d) Handling Class Imbalance with SMOTE
- **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to address class imbalance. The classifiers were retrained, and results were compared with those from the imbalanced data.

### (e) Extra Practice: Classifier Chains and Evaluation Metrics
- **Classifier Chain Method**: The method was researched and applied to the problem.
- **Confusion Matrices, Precision, Recall, ROC, and AUC** were calculated to evaluate the multi-label classifiers.

## Task 2: K-Means Clustering on Multi-Class and Multi-Label Data

### (a) K-Means Clustering
- **K-Means clustering** was applied to the entire dataset. The optimal number of clusters, `k`, was determined using **Gap Statistics** and **Silhouette scores**.
  
### (b) Label Assignment
- For each cluster, the majority label triplet (Family, Genus, Species) was determined by reading the true labels.
  
### (c) Hamming Distance, Hamming Score, and Hamming Loss
- The **Hamming distance, score, and loss** were calculated between the true labels and the labels assigned by the clusters.
  
## How to Run

### Requirements

- Python 3.x
- Jupyter Notebook
- Required Python libraries (can be installed via `pip`):
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `imbalanced-learn`
  - `seaborn`

### Instructions

1. Clone the repository and navigate to the project folder.
2. Open the Jupyter Notebook (`Huang_Bor-Sheng.ipynb`).
3. Run the notebook cells to execute the tasks and view the results.

## Results

- **SVM Classifiers**: Reported metrics include exact match score, Hamming score, and Hamming loss for both imbalanced and SMOTE-processed data.
- **K-Means Clustering**: The optimal number of clusters was found, and the performance of clustering was evaluated using Hamming distance, score, and loss.

## License

This project is intended for academic purposes and is based on the DSCI 552 course material.

