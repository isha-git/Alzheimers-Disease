import sklearn
import scipy.io as sio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC # "Support Vector Classifier"
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import classification_report

# svm.csv is the dataset file.
dataset = pd.read_csv('svm.csv')

# Print the featurs of the dataset along with first five rows as a sample
print(dataset.head())

#G roup feather contains the final class to be predicted, and hence is removed from the input feature set
data = dataset.drop('Group', axis=1)

# Gene name contains the name of the corresponding gene, and is not required for training purposes
data = data.drop('Gene Name', axis=1)

# Missing data is replaced by the number 0 for consistency
data = data.replace(np.NaN, 0)
X = data[2:]
print(dataset.head())

# This is the output class, and contrains the final Group column which was removed in the above steps from the input features
datay = dataset['Group']
y = datay[2:]

# The dataset is shuffled and divided into 67% train and 33% test set.
train, test, train_labels, test_labels = train_test_split(X,y,test_size=0.33,random_state=2)

# Initialize our classifier
# Create Decision Tree classifer object
# Uncomment the command according to the classifier needed
# We obtained the best accuracy for Decision Trees (88.29%)

clf = SVC(kernel='rbf', C=2)        # Training using Support Vector Classifier with Radial kernel and regularization constant = 2
# clf = SVC(kernel = 'linear')        # Training using Support Vector Classifier with Linear kernel and regularization constant = 2
# clf = SVC(kernel = 'poly')          # Training using Support Vector Classifier with Polynomial kernel and regularization constant = 2
# clf = SVC(kernel = 'sigmoid')       # Training using Support Vector Classifier with Sigmoid kernel and regularization constant = 2
# clf = DecisionTreeClassifier()      # Training using Decision Tree Classifier

# Fit the chosen classifier on the input dataset
clf = clf.fit(train,train_labels)

# Predict the response for test dataset
y_pred = clf.predict(test)

print(y_pred)

# Checking the model accuracy
print("Accuracy:",accuracy_score(test_labels, y_pred))
print(classification_report(test_labels, y_pred))
