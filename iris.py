import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from warnings import filterwarnings
filterwarnings(action='ignore')


# In[3]:


iris=pd.read_csv("iris.csv")
print(iris)


# In[4]:


print(iris.shape)


# In[5]:


print(iris.describe())


# In[6]:


#Checking for null values
print(iris.isna().sum())
print(iris.describe())


# In[7]:


iris.head()


# In[8]:


iris.head(150)


# In[9]:


iris.tail(100)


# In[10]:


n = len(iris[iris['Species'] == 'versicolor'])
print("No of Versicolor in Dataset:",n)


# In[11]:


n1 = len(iris[iris['Species'] == 'virginica'])
print("No of Virginica in Dataset:",n1)


# In[12]:


n2 = len(iris[iris['Species'] == 'setosa'])
print("No of Setosa in Dataset:",n2)


# In[13]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Versicolor', 'Setosa', 'Virginica']
s = [50,50,50]
ax.pie(s, labels = l,autopct='%1.2f%%')
plt.show()


# In[14]:


#Checking for outliars
import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([iris['Sepal.Length']])
plt.figure(2)
plt.boxplot([iris['Sepal.Width']])
plt.show()


# In[15]:


iris.hist()
plt.show()


# In[16]:


iris.plot(kind ='density',subplots = True, layout =(3,3),sharex = False)





iris.plot(kind ='box',subplots = True, layout =(2,5),sharex = False)





plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='Petal.Length',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='Petal.Width',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='Sepal.Length',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='Sepal.Width',data=iris)





sns.pairplot(iris,hue='Species');


# In[20]:


#Heat Maps
fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.heatmap(iris.corr(),annot=True,cmap='cubehelix',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)


# In[19]:


X = iris['Sepal.Length'].values.reshape(-1,1)
print(X)


# In[20]:


Y = iris['Sepal.Width'].values.reshape(-1,1)
print(Y)


# In[21]:


plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(X,Y,color='b')
plt.show()


# In[22]:


#Correlation 
corr_mat = iris.corr()
print(corr_mat)


# In[23]:


from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[24]:


train, test = train_test_split(iris, test_size = 0.25)
print(train.shape)
print(test.shape)


# In[25]:


train_X = train[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
                 'Petal.Width']]
train_y = train.Species

test_X = test[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
                 'Petal.Width']]
test_y = test.Species


# In[26]:


train_X.head()


# In[27]:


test_y.head()


# In[28]:


test_y.head()


# In[29]:


#Using LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_y))


# In[30]:


#Confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,prediction))


# In[31]:


#Using Support Vector
from sklearn.svm import SVC
model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

from sklearn.metrics import accuracy_score
print("Acc=",accuracy_score(test_y,pred_y))


# In[32]:


#Using KNN Neighbors
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred2))


# In[33]:


#Using GaussianNB
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(train_X,train_y)
y_pred3 = model3.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred3))


# In[34]:


#Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(train_X,train_y)
y_pred4 = model4.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred4))



results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines', 'Naive Bayes','KNN' ,'Decision Tree'],
    'Score': [0.947,0.947,0.947,0.947,0.921]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)




##saving model 
import pickle


# save the model to disk-2
filename2 = 'finalized_model2'
pickle.dump(model2, open(filename2, 'wb'))

# load the model from disk
loaded_model2 = pickle.load(open(filename2, 'rb'))

##testing 
y_pred12 = loaded_model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score of deployed model2:",accuracy_score(test_y,y_pred12))
