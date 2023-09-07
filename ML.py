from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


names = ['hourly', 'overtime', 'annual', 'level']
dataset = read_csv('pay rate.csv', names=names)

# print(dataset.shape) #check if the data load properly 
# print(dataset.head(50)) #return the first 50 row of data

# splits the dataset into training and validation sets 
array = dataset.values
X = array[:, 0:4]
Y = array[:,3]
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y, test_size=0.20, random_state=1) #80 percent for for training and 20% are for validation

model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
prediction = model.predict(X_validation)
print(accuracy_score(Y_validation, prediction))
print(confusion_matrix(Y_validation, prediction))
print(classification_report(Y_validation, prediction))