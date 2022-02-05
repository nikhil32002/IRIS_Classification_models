from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading the dataset 
iris=pd.read_csv("iris.csv")
train, test = train_test_split(iris, test_size = 0.25)
print(train.shape)
print(test.shape)
#
train_X = train[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
                 'Petal.Width']]
train_y = train.Species

test_X = test[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
                 'Petal.Width']]
test_y = test.Species

# load the model from disk-1
import pickle
filename = 'finalized_model.sav'
loaded_model1 = pickle.load(open(filename, 'rb'))

##testing 
y_pred11 = loaded_model1.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score of deployed model (loaded_model1):",accuracy_score(test_y,y_pred11))
