#!/usr/bin/python3

import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

classif = np.memmap(sys.argv[1], dtype=np.float32, mode='c', order ='C')
exact = np.memmap(sys.argv[2], dtype=np.float32, mode='r', order ='C')

renum = []
ind0 = np.where(classif == 0);
t,_ = np.histogram(exact[ind0], [-1.5, -0.5, 0.5, 1.5])
c0 = t.argmax() - 1
ind1 = np.where(classif == 1);
t,_ = np.histogram(exact[ind1], [-1.5, -0.5, 0.5, 1.5])
c1 = t.argmax() - 1
ind2 = np.where(classif == 2);
t,_ = np.histogram(exact[ind2], [-1.5, -0.5, 0.5, 1.5])
c2 = t.argmax() - 1

print("[" , c0 , ", ", c1, ", ", c2 ,"]")
classif[ind0] = c0
classif[ind1] = c1
classif[ind2] = c2


print("Accuracy: ", accuracy_score(exact, classif))
print("Precision: ", precision_score(exact, classif, average = 'macro'))
print("Recall: ", recall_score(exact, classif, average = 'macro'))
print("Confusion Matrix: ", confusion_matrix(exact, classif))
