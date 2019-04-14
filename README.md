# Fake-news-detection
Fake News Detection on Liar dataset


## DataPrep.py:
for preprecessing the metadata and the text data. 

>*Processing the metadata*

## FeatureSelection.py:
contains utility functions for term doc matrix, tfidf vectorizer, POS tagging, glove model and confusion matrix to be used in the other files

## ml_project_classifiers.py:
contains classifiers like lienar svm, svm with sgd, naive-bayes, random forest and logistic regression for classifying the data

## cnn_model.py and text_cnn.py:
contains the code for cnn model implementation on the text data.


### Checklist:
- [x] process metadata 
- [x] process text data
- [ ] if time permits, dominant feature selection through svd
- [ ] XGBoost on metadata
- [x] CNN model 
- [ ] LSTM model
- [ ] Combine the best models
