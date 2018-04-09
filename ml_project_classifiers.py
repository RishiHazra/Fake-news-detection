# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 14:48:17 2018

@author: Rishi
"""
import DataPrep
import FeatureSelection
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score



#========================= Classifying the text data ===============================#

#building classifier using naive bayes 
nb_pipeline = Pipeline([
        ('NBCV',FeatureSelection.countV),
        ('nb_clf',MultinomialNB())])

nb_pipeline.fit(DataPrep.train_data[2],DataPrep.train_data[1])
predicted_nb = nb_pipeline.predict(DataPrep.test_data[2])
np.mean(predicted_nb == DataPrep.test_data[2])


#building classifier using logistic regression
logR_pipeline = Pipeline([
        ('LogRCV',FeatureSelection.countV),
        ('LogR_clf',LogisticRegression())
        ])

logR_pipeline.fit(DataPrep.train_data[2],DataPrep.train_data[1])
predicted_LogR = logR_pipeline.predict(DataPrep.test_data[2])
np.mean(predicted_LogR == DataPrep.test_data[1])


#building Linear SVM classfier
svm_pipeline = Pipeline([
        ('svmCV',FeatureSelection.countV),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline.fit(DataPrep.train_data[2],DataPrep.train_data[1])
predicted_svm = svm_pipeline.predict(DataPrep.test_data[2])
np.mean(predicted_svm == DataPrep.test_data[1])


#using SVM Stochastic Gradient Descent on hinge loss
sgd_pipeline = Pipeline([
        ('svm2CV',FeatureSelection.countV),
        ('svm2_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))
        ])

sgd_pipeline.fit(DataPrep.train_data[2],DataPrep.train_data[1])
predicted_sgd = sgd_pipeline.predict(DataPrep.test_data[2])
np.mean(predicted_sgd == DataPrep.test_data[1])


#random forest
random_forest = Pipeline([
        ('rfCV',FeatureSelection.countV),
        ('rf_clf',RandomForestClassifier(n_estimators=200,n_jobs=3))
        ])
    
random_forest.fit(DataPrep.train_data[2],DataPrep.train_data[1])
predicted_rf = random_forest.predict(DataPrep.test_data[2])
np.mean(predicted_rf == DataPrep.test_data[1])


# predicting using 5-Fold cross validation
def build_confusion_matrix(classifier):
    
    k_fold = KFold(n=len(DataPrep.train_data), n_folds=5)
    scores = []
    confusion = np.zeros((6,6))

    for train_ind, test_ind in k_fold:
        train_text = DataPrep.train_data.iloc[train_ind][2] 
        train_y = DataPrep.train_data.iloc[train_ind][1]
    
        test_text = DataPrep.train_data.iloc[test_ind][2]
        test_y = DataPrep.train_data.iloc[test_ind][1]
        
        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)
        
        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(list(test_y),predictions,average='macro')
        scores.append(score)
    
    return (print('Total statements classified:', len(DataPrep.train_data)),
    print('Score:', sum(scores)/len(scores)),
    print('score length', len(scores)),
    print('Confusion matrix:'),
    print(confusion))
    
#K-fold cross validation for all classifiers
    
build_confusion_matrix(nb_pipeline)
# 0.2037045113531469
build_confusion_matrix(logR_pipeline)
# 0.22491178360259526
build_confusion_matrix(svm_pipeline)
# 0.23150749974860907
build_confusion_matrix(sgd_pipeline)
# 0.20946921834197516
build_confusion_matrix(random_forest)
# 0.2182263314779121



#========================= Classifying the metadata ===============================#

from DataPrep import data_return

X_data, class_names= data_return()
X_data = np.nan_to_num(X_data)
clf = SGDClassifier()
model=clf.fit(X_data, DataPrep.train_data[1])

# TODO: This has to be done on test_data
op=model.predict(X_data[None,0,:])

accuracy=0
Y_pred={}
for i in range(len(DataPrep.train_data[1])):
    accuracy+=int(model.predict(X_data[None,i,:])== DataPrep.train_data[1][i])
    Y_pred[i]=model.predict(X_data[None,i,:])[0]
accuracy/=len(DataPrep.train_data[1])


cm = confusion_matrix(DataPrep.train_data[1],list(Y_pred.values()))

plt.figure()
confusion_matrix=FeatureSelection.plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')



