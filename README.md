# Fake-news-detection
Fake News Detection on Liar dataset


## DataPrep.py:
for preprecessing the metadata and the text data. 

**@sourabh :bowtie:- please take care of the multilabel binarizer in this code for the test data. you will find it under the heading**
>*Processing the metadata*

## FeatureSelection.py:
contains utility functions for term doc matrix, tfidf vectorizer, POS tagging, glove model and confusion matrix to be used in the other files

## ml_project_classifiers.py:
contains classifiers like lienar svm, svm with sgd, naive-bayes, random forest and logistic regression for classifying the data

## cnn_model.py and text_cnn.py:
contains the code for cnn model implementation on the text data.
