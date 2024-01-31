import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("..\\data.csv")

df.drop(['Unnamed: 32', 'id'], axis = 1, inplace=True)

def diagnosis_value(diagnosis):
    if diagnosis == 'M':
        return 1
    else:
        return 0
    
df['diagnosis'] = df['diagnosis'].apply(diagnosis_value)


sns.lmplot(x = 'radius_mean', y = 'texture_mean',
           hue = 'diagnosis', data = df)
plt.show()

sns.lmplot(x = 'smoothness_mean', y = 'compactness_mean',
           data = df, hue = 'diagnosis')
plt.show()

X = np.array(df.iloc[:, 1:])
y = np.array(df['diagnosis'])

from sklearn.model_selection import cross_val_score, train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.33, random_state = 42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 13)
knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))

neighbors = []
cv_scores = []

from sklearn.model_selection import cross_val_score
for k in range(1, 51, 2):
    neighbors.append(k)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(
        knn, X_train, y_train, cv=10, scoring = 'accuracy')
    cv_scores.append(scores.mean())
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print('The optimal number of neighbors is % d ' % optimal_k)

plt.figure(figsize=(10,6))
plt.plot(neighbors, MSE)
plt.xlabel('Number of neighbors')
plt.ylabel('Misclassification Error')
plt.show()