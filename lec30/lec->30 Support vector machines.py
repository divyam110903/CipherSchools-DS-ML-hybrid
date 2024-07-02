# /*SVM's
# CLASSIFY DATA POINTS WE CREATE A DECISION MARGIN A LINE
# SVM'S ARE A CLASS OF LINEAR CLASSIFIERS
# bw support margins we draw deceison margin more the gap it is easier
# 3 PLANES  MAX MARGIN HYPERPLANE
# MAXIMUM MARGIN
# POSITIVE HYPER PLANE


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#loading data
iris=load_iris()
x=iris.data
y=iris.target


#split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#model
model=SVC()
model.fit(x_train,y_train)
#prediction
y_pred=model.predict(x_test)
#accuracy
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)