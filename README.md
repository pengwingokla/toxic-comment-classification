# 🧠 Toxic Comment Classification
This project focuses on building a machine learning pipeline to classify toxic comments in online platforms. Leveraging natural language processing (NLP) techniques, the model identifies various types of toxicity—such as hate speech, threats, and identity-based attacks—within user-generated content.

## 🔍 Features
* TF-IDF vectorization for feature extraction from raw text
* Supervised learning models including Logistic Regression, SVM, and Random Forest
* Threshold tuning for optimizing precision-recall tradeoffs on imbalanced data
* Evaluation metrics: F1-score, precision, recall, ROC-AUC
* Clean, reproducible pipeline using scikit-learn and pandas

## 🗃️ Dataset
The dataset from the Jigsaw Toxic Comment Classification Challenge consists of a large number of Wikipedia comments labeled for various types of toxic behavior. These labels include: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
It is used to create a model that predicts the probability of each type of toxicity in a given comment. The dataset includes the following files:
* train.csv: Contains the training set with comments and their binary labels.
* test.csv: The test set, where the model predicts toxicity probabilities.
* sample_submission.csv: A sample submission file in the correct format.
* test_labels.csv: Contains labels for the test data.

This dataset is intended for use in toxic comment classification tasks, where identifying harmful content in user-generated comments is crucial for online community moderation.

Preprocessing includes tokenization, lowercasing, and stopword removal.

## 📈 Goal
To accurately detect toxic behavior in text and contribute to safer online communities by developing interpretable, scalable models suitable for integration into moderation tools.
