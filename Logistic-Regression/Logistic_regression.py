#The scikit-learn (sklearn) library in Python is a powerful open-source toolkit for machine learning. 
# It provides ready-to-use algorithms and utilities for tasks
# like classification, regression, clustering, dimensionality reduction, preprocessing, and model evaluation. 
# In short, it’s the go-to library for building and experimenting with ML models efficiently.
from sklearn.datasets import load_breast_cancer  #helps to call dataset of breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split   #this is used for divide the dataset in training and testing data
from sklearn.metrics import accuracy_score 


x,y = load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=23)

clf = LogisticRegression(max_iter = 10000,random_state=0)
clf.fit(x_train,y_train)

acc = accuracy_score(y_test,clf.predict(x_test))*100

print(f"Logisic model accuracy is {acc:.2f}%8")
