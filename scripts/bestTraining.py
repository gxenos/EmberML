import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import graphviz
import pickle
from time import time
import json

#---------- READ FROM FILES ----------
# Read from .dat vectorized arrays to NumPy  memory-maps
train_size = 100000
test_size = 40000
columns = 2351

path = "../data"

x_train = np.memmap(path+"/X_train.dat", dtype=np.float32, mode="r", shape=(train_size, 2351))
y_train = np.memmap(path+"/y_train.dat", dtype=np.float32, mode="r", shape=train_size)

x_validation = x_train[70000:90000]
y_validation = y_train[70000:90000]

x_train = x_train[:60000]
y_train = y_train[:60000]

x_test = np.memmap(path+"/X_test.dat", dtype=np.float32, mode="r", shape=(test_size, 2351))
y_test = np.memmap(path+"/y_test.dat", dtype=np.float32, mode="r", shape=test_size)

data = {}
data['models'] = []

#---------- TRAIN SINGLE DECISION TREE----------
print('-' * 30)
print('Single Decision Tree')

model = tree.DecisionTreeClassifier(min_samples_leaf=2)

t = time()
model.fit(x_train, y_train)
t = round(time() - t, 3)
accuracy = model.score(x_test, y_test)
r2_accuracy = r2_score(model.predict(x_test), y_test)
matrix = confusion_matrix(y_test, model.predict(x_test))
filename = 'single_tree.sav'
pickle.dump(model, open(filename, 'wb'))
data['models'].append({
    'name': 'decision_tree',
    'train_time': t,
    'accuracy': accuracy,
    'r2_accuracy': r2_accuracy,
    'confusion_matrix': matrix.tolist(),
})
tree.export_graphviz(model, out_file='decision_tree.dot')
print(f'Training Time: {t} seconds.')
print(f'Accuracy on test set: {accuracy}')
print(f'R squared on test set: {r2_accuracy}')
print(f'Confusion Matrix: {matrix[0]}, {matrix[1]}')
print('-' * 30)


# ---------- TRAIN RANDOM FOREST----------
print('Random Forest')

model = RandomForestClassifier(n_jobs=-1, n_estimators=500, max_features=0.2, min_samples_leaf=1)

t = time()
model.fit(x_train, y_train)
t = round(time() - t, 3)
accuracy = model.score(x_test, y_test)
r2_accuracy = r2_score(model.predict(x_test), y_test)
matrix = confusion_matrix(y_test, model.predict(x_test))
filename = 'random_forest.sav'
pickle.dump(model, open(filename, 'wb'))
data['models'].append({
    'name': 'random_forest',
    'train_time': t,
    'accuracy': accuracy,
    'r2_accuracy': r2_accuracy,
    'confusion_matrix': matrix.tolist(),
})
print(f'Training Time: {t} seconds.')
print(f'Accuracy on test set: {accuracy}')
print(f'R squared on test set: {r2_accuracy}')
print(f'Confusion Matrix: {matrix[0]}, {matrix[1]}')
print('-' * 30)


with open('models_meta.json', 'w') as outfile:  
    json.dump(data, outfile)