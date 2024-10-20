# README for Heart Disease Prediction Model


## Description: 
This project builds and evaluates multiple machine learning models to predict heart disease based on various patient features. The dataset used for this analysis contains patient health metrics and the target variable (HeartDisease), indicating whether the patient has heart disease (1) or not (0). The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and final prediction using various classifiers.

## Project Structure:
Data Analysis and Preprocessing

Exploratory Data Analysis (EDA)

Model Implementation

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVM)

Evaluation Metrics

Model Prediction

## Data Analysis and Preprocessing
The dataset (heart.csv) is loaded using pandas, and the basic information about the dataset is printed to understand its structure and ensure there are no missing values.

## Data Overview:

The dataset has 918 entries and 12 columns with no missing values.

The columns include features like Age, Sex, ChestPainType, RestingBP, Cholestrol, and HeartDisease.

## Exploratory Data Analysis (EDA)

A Correlation Matrix is created using seaborn to visualize the correlations between different features.

A Histogram is plotted for each feature to understand the distribution of values across the dataset.

## Train-Test Split

The data is split into training and test sets with 75% for training and 25% for testing using train_test_split from scikit-learn.

## Model Implementation

**Logistic Regression:**

A logistic regression model is trained, and a confusion matrix is plotted to visualize the classification results.
Evaluation Metrics: Testing Accuracy, Sensitivity, Specificity, and Precision are computed.

**Decision Tree Classifier:**

A decision tree model is trained with max_depth=5 using the entropy criterion.
Cross-validation is used to evaluate model accuracy.
The model is evaluated using a confusion matrix and metrics like precision, recall, and F1-score.

**Random Forest Classifier:**

A random forest model with 500 estimators is trained.
The model’s accuracy, sensitivity, specificity, and precision are computed and displayed along with a classification report.

**Support Vector Machine (SVM):**

An SVM model with a linear kernel is trained.
The model’s performance is evaluated using a confusion matrix, and metrics such as accuracy, sensitivity, and precision are printed.

**Evaluation Metrics**
The following metrics are computed for each model:

**Accuracy:** Measures the proportion of correctly predicted instances over the total instances.
**Sensitivity (Recall):** The true positive rate, indicating how well the model identifies positive cases.
**Specificity:** The true negative rate, indicating how well the model identifies negative cases.
**Precision:** The proportion of true positive instances over all instances predicted as positive.

## Model Prediction
The final model implementation takes patient features as input and predicts whether the patient has heart disease.

## Example Inputs:
An input of (40,0,0,140,289,0,0,172,0,0,0) is predicted as "The patient seems to be Normal".

Another input of (62,0,140,268,0,0,160,0,3.6,0,1) is predicted as "The patient seems to have heart disease".

## Installation and Setup

**Clone the repository:**
bash
Copy code
git clone <repository-url>
Install the necessary libraries:
bash
Copy code
pip install numpy pandas matplotlib seaborn scikit-learn
Run the notebook or script to train the models and test the predictions:
bash
Copy code
python heart_disease_prediction.py

## Dependencies
Python (3.x)

**NumPy:** For numerical operations
**Pandas:** For data manipulation and analysis
**Matplotlib and Seaborn:** For data visualization
**scikit-learn:** For model implementation and evaluation
**Notes:**

Ensure that the dataset heart.csv is in the working directory.

The Logistic Regression model may raise a ConvergenceWarning. You can either increase max_iter or scale the data to address this issue as per the scikit-learn documentation.

## Conclusion
This project provides a comprehensive approach to predicting heart disease using different machine learning models. By comparing their performance, it helps to identify which model offers the best accuracy and reliability for this specific task.
