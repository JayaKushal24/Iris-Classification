import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] # As per the iris dataset information

# Load the data
df = pd.read_csv('iris.data', names=columns)

df.head()

df.describe()

sns.pairplot(df, hue='Class_labels')  

data = df.values
X = data[:,0:4]
Y = data[:,4]

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Calculate avarage of each features for all classes
Y_Data = np.array([np.average(X[:, i][Y_encoded ==j].astype('float32')) for i in range (X.shape[1]) for j in (np.unique(Y_encoded ))])
print('Y_Data',Y_Data)
Y_Data_reshaped = Y_Data.reshape(4, 3)
print(Y_Data_reshaped)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns)-1)
width = 0.2
print(Y_Data_reshaped)
print(X_axis)


plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y_encoded , test_size=0.2)

from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)

# Predict from the test dataset
predictions = svn.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

X_new = np.array([
    [5.7, 3.0, 4.2, 1.2],
    [6.4, 2.8, 5.6, 2.1],
    [4.8, 3.4, 1.9, 0.2],
    [6.0, 2.2, 4.0, 1.0],
])


prediction = svn.predict(X_new)
prediction= label_encoder.inverse_transform(prediction)
print("Prediction of Species: {}".format(prediction))