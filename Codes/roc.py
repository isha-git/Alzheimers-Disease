import sklearn
import scipy.io as sio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.svm import SVC # "Support Vector Classifier"
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
import pylab as pl
import csv

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

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

# There are 4 output classes- C1, C2, C3, and C4
# C1-AD: probable pathogenic genes
# C2-AD: high confidence genes
# C3-AD: related genes, and
# C4-AD: possibly associated genes.
y = label_binarize(y, classes=[1, 2,3,4])

n_classes = y.shape[1]
lw=2

# Add noise to make the model more robust
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# Shuffle and split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the others
# Here, Linear kernel is used for Support Vector Machines. However, we can change the kernel, or also use Decision Tree Classifies. Refer to classifier.py for their commands
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))

# Fit the chosen classifier on the input dataset
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute the Receiver operating characteristic (ROC) curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute Area Unver the Curve (AUC)
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()


colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
