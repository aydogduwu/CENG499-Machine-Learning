import pickle
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt


dataset, labels = pickle.load(open("../data/part2_dataset1.data", "rb"))

# create 4 SVMs with different kernels and C values
svm_linear_c1 = svm.SVC(kernel='linear', C=1)
svm_linear_c5 = svm.SVC(kernel='linear', C=5)
svm_rbf_c1 = svm.SVC(kernel='rbf', C=1)
svm_rbf_c5 = svm.SVC(kernel='rbf', C=5)

# train the SVMs
svm_linear_c1.fit(dataset, labels)
svm_linear_c5.fit(dataset, labels)
svm_rbf_c1.fit(dataset, labels)
svm_rbf_c5.fit(dataset, labels)

# draw the decision boundaries for each SVM
x_min, x_max = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
y_min, y_max = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

titles = ['SVM with Linear Kernel, C=1',
            'SVM with Linear Kernel, C=5',
            'SVM with RBF Kernel, C=1',
            'SVM with RBF Kernel, C=5']

for i, clf in enumerate((svm_linear_c1, svm_linear_c5, svm_rbf_c1, svm_rbf_c5)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, edgecolors='k', cmap=plt.cm.coolwarm)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.savefig('svm_dataset1.png')
plt.show()



