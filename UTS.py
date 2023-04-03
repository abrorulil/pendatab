#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd

# Load dataset
url = 'https://raw.githubusercontent.com/datasets/breast-cancer/master/data/breast-cancer.csv'
df = pd.read_csv(url)


# Preprocessing
df['class'] = df['class'].map({'no-recurrence-events': 0, 'recurrence-events': 1})
df.head()


# In[15]:


from sklearn.model_selection import train_test_split

# Split dataset into training and testing data
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[36]:


from sklearn.preprocessing import LabelEncoder

X_train['falsede-caps'] = X_train['falsede-caps'].replace({True: 'True', False: 'False'})
X_test['falsede-caps'] = X_test['falsede-caps'].replace({True: 'True', False: 'False'})

# Encoding categorical data
le = LabelEncoder()
X_train['mefalsepause'] = le.fit_transform(X_train['mefalsepause'])
X_test['mefalsepause'] = le.transform(X_test['mefalsepause'])

X_train['falsede-caps'] = le.fit_transform(X_train['falsede-caps'].fillna('no'))
X_test['falsede-caps'] = le.transform(X_test['falsede-caps'].fillna('no'))

X_train['breast-quad'] = le.fit_transform(X_train['breast-quad'].fillna('unknown'))
X_test['breast-quad'] = le.transform(X_test['breast-quad'].fillna('unknown'))

X_train['irradiat'] = le.fit_transform(X_train['irradiat'])
X_test['irradiat'] = le.transform(X_test['irradiat'])

X_train['age'] = pd.to_numeric(X_train['age'], errors='coerce')
X_test['age'] = pd.to_numeric(X_train['age'], errors='coerce')
X_train['tumor-size'] = pd.to_numeric(X_train['tumor-size'], errors='coerce')
X_test['tumor-size'] = pd.to_numeric(X_train['tumor-size'], errors='coerce')
X_train['inv-falsedes'] = pd.to_numeric(X_train['inv-falsedes'], errors='coerce')
X_test['inv-falsedes'] = pd.to_numeric(X_train['inv-falsedes'], errors='coerce')


# In[37]:


from sklearn.preprocessing import StandardScaler

# Feature scaling
sc = StandardScaler()
X_train[['age', 'tumor-size', 'inv-falsedes', 'deg-malig']] = sc.fit_transform(X_train[['age', 'tumor-size', 'inv-falsedes', 'deg-malig']])
X_test[['age', 'tumor-size', 'inv-falsedes', 'deg-malig']] = sc.transform(X_test[['age', 'tumor-size', 'inv-falsedes', 'deg-malig']])


# In[39]:


from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)

nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb)
nb_recall = recall_score(y_test, y_pred_nb)

print('Naive Bayes:')
print(f'Accuracy: {nb_accuracy:.3f}')
print(f'Precision: {nb_precision:.3f}')
print(f'Recall: {nb_recall:.3f}\n')

# KNN
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn)
knn_recall = recall_score(y_test, y_pred_knn)

print('K-Nearest Neighbors:')
print(f'Accuracy: {knn_accuracy:.3f}')
print(f'Precision: {knn_precision:.3f}')
print(f'Recall: {knn_recall:.3f}')


# In[ ]:




