#PREDICTING BINARY OUTCOMES

#Logistic Regression
#Logistic Regression is a classification algorithm that is used to assign observations to a discrete set of classes

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#loading iris data
iris = load_iris()
x=iris.data
y=iris.target

#using  2 classes only  for binary classification
x=x[y!=2]
y=y[y!=2]

#splitting of data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)  

#training of model
model=LogisticRegression()
model.fit(x_train,y_train)

#predictions
y_pred =model.predict(x_test)
#evaluation
print("accuracy")
print(accuracy_score(y_test,y_pred))
print(y_test)
print(y_pred)