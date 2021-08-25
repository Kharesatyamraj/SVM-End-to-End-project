# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 09:09:37 2021

@author: khare
"""


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


#Import Dataset
train1= pd.read_csv(r"C:/Users/khare/Downloads\IRIS.csv")

###Preprocessing And EDA

## data preprocessing

# Duplicate Value check

#train dataset
train1.duplicated().sum() # 5982 duplicates found
train1.drop_duplicates(keep="first",inplace=True)# removing duplicates


#traindata set preprocessing
train1.head()

train1.isna().sum()  # no missing values



#EDA

#train Dataset
Eda = train1.describe()
Skewness=train1.skew()
Kurtosis= train1.kurt()

##Visulization

#Pair Plot
sns.pairplot(train1,hue="species")#Pairplot of train1 dataset


# Count Plot
sns.countplot(x="species",data=train1)# Countplot of train1 dataset 


# train and test dataset split into train_X,train_Y,test_X and test_Y
from sklearn.model_selection import train_test_split
X = train1.iloc[:,:-1]
X.head()
Y = train1.iloc[:, -1]
Y.head()

train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.3,random_state = 0)


### Normalization
def norm_func(i):
	x = (i-i.min())	/(i.max()-i.min())
	return(x)

# Normalization of train Dataset

train_X= norm_func(train_X)

# Normalization of test Dataset

test_X= norm_func(test_X)


#Model Building with different Kernels

# kernel = linear

model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_Y)


#SVC applied on test dataset
pred_test_linear = model_linear.predict(test_X)

#Accuracy 
accuracy1=np.mean(pred_test_linear == test_Y)

# Evaluation-Matrix
mat1=pd.crosstab(pred_test_linear, test_Y)

#SVC applied on train dataset
pred_train_linear = model_linear.predict(train_X)

#Accuracy 
accuracy2=np.mean(pred_train_linear == train_Y)

# Evaluation-Matrix
mat2=pd.crosstab(pred_train_linear, train_Y)

### kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X, train_Y)


#SVC applied on test dataset
pred_test_poly = model_poly.predict(test_X)

#Accuracy 
accuracy3=np.mean(pred_test_poly==test_Y)

# Evaluation-Matrix
mat3=pd.crosstab(pred_test_poly, test_Y)

#SVC applied on train dataset
pred_train_poly = model_poly.predict(train_X)

#Accuracy 
accuracy4=np.mean(pred_train_poly == train_Y)

# Evaluation-Matrix
mat4=pd.crosstab(pred_train_poly, train_Y)


### kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_Y)


#SVC applied on test dataset
pred_test_rbf = model_rbf.predict(test_X)

#Accuracy 
accuracy5=np.mean(pred_test_rbf==test_Y)

# Evaluation-Matrix
mat5=pd.crosstab(pred_test_rbf, test_Y)

#SVC applied on train dataset
pred_train_rbf = model_rbf.predict(train_X)

#Accuracy 
accuracy6=np.mean(pred_train_rbf == train_Y)

# Evaluation-Matrix
mat6=pd.crosstab(pred_train_rbf, train_Y)


### kernel = precomputed
model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(train_X, train_Y)


#SVC applied on test dataset
pred_test_sigmoid= model_sigmoid.predict(test_X)

#Accuracy 
accuracy7=np.mean(pred_test_sigmoid==test_Y)

# Evaluation-Matrix
mat7=pd.crosstab(pred_test_rbf, test_Y)

#SVC applied on train dataset
pred_train_sigmoid = model_sigmoid.predict(train_X)

#Accuracy 
accuracy8=np.mean(pred_train_sigmoid== train_Y)

# Evaluation-Matrix
mat8=pd.crosstab(pred_train_sigmoid, train_Y)

## Accuracy Table 

# Accuracy table to compare train and test accuracies of different kernels and select the right fitted with highest accuracy model. 
dic = {"Kernels":pd.Series(["linear", "poly", "rbf", "sigmoid" ]),"Test Accuracy":pd.Series([accuracy1, accuracy3, accuracy5, accuracy7]),"Train Accuracy":pd.Series([accuracy2,accuracy4,accuracy6,accuracy8])}
accuracy_table = pd.DataFrame(dic)
accuracy_table

# Since rbf model has the right fit ,we apply rbf kernel
#   Kernels  Test Accuracy  Train Accuracy
#0   linear       0.955556        0.960784
#1     poly       1.000000        0.980392
#2      rbf       0.955556        0.960784
#3  sigmoid       0.466667        0.352941

#create a pickle file using serialization
#import pickle
#pickle_out = open("model_rbf.pkl","wb")
#pickle.dump(model_rbf,pickle_out)
#pickle_out.close()


