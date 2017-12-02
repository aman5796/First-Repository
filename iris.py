from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

clf1 = GaussianNB()
clf1.fit(X_train,Y_train)
pred = clf1.predict(X_test)
score1 = clf1.score(X_test,Y_test)
print("Accuracy of naive bayes is  : ")
print(score1)
print("\n")


clf2 = svm.SVC(kernel='rbf')
clf2.fit(X_train,Y_train)
pred = clf2.predict(X_test)
score2 = clf2.score(X_test,Y_test)
print("Accuracy of SVM is  : ")
print(score2)
print("\n")
