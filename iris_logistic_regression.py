import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV

# loading iris dataset into memory
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue='species')

#Seperating dependent and independent variable
X=iris.values[:,:4]
y=iris.values[:,4]
#Dividing dataset into training and testing
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.5,random_state=1)

#creating a object of logisticregression
model=LogisticRegressionCV()

#fitting model with training data
model.fit(X_train,y_train)

print ("Accuracy={:.2f}".format(model.score(X_test,y_test)))
