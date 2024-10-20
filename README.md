# SMS Spam Classifier

This project implements a machine learning model to classify SMS messages as spam or not spam. The model is trained using a labeled dataset of SMS messages and can predict whether a new message is spam or legitimate based on the text.

## Project Structure

- `sms_spam_classifier.ipynb`: The Jupyter Notebook containing the code for preprocessing, training, and evaluating the SMS spam classifier model.
- `data/`: The folder containing the dataset used for training and testing.
- `models/`: Folder to save trained models.
- `README.md`: This file providing an overview of the project.

## Features

- **Data Preprocessing**: Tokenization, stopword removal, and vectorization (e.g., TF-IDF) to prepare the SMS messages for training.
- **Model Training**: Trained with various models such as Naive Bayes, Support Vector Machines (SVM), and Logistic Regression.
- **Evaluation**: Precision, recall, F1-score, and accuracy metrics are used to evaluate model performance.
- **Model Inference**: A simple function that allows classifying new SMS messages as spam or not spam.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python libraries:
  - pandas
  - scikit-learn
  - nltk
  - matplotlib

Install dependencies using:
```bash
pip install pandas scikit-learn nltk matplotlib
