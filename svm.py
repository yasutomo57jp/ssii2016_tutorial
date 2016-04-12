from sklearn.datasets import fetch_mldata
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

mnist = fetch_mldata("MNIST original", data_home=".")
data = np.asarray(mnist.data, np.float32)
data_train, data_test, label_train, label_test = train_test_split(data, mnist.target, test_size=0.2)

classifier = LinearSVC()
classifier.fit(data_train, label_train)

result = classifier.predict(data_test)

cmat = confusion_matrix(label_test, result)
print(cmat)
