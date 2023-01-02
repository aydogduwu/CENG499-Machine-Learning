import pickle
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


dataset, labels = pickle.load(open("../data/part2_dataset2.data", "rb"))

# preprocess the data by using StandardScaler
scaler = StandardScaler()
preprocessed = scaler.fit_transform(dataset)

params = {'kernel': ['linear', 'rbf'], 'C': [5, 10, 50, 100, 500, 1000]}

clfs = []

for i in range(5):
    # create a 10-fold cross validation
    skf = StratifiedKFold(n_splits=10, shuffle=True)

    # create an SVM classifier
    svm_clf = svm.SVC()

    # use GridSearchCV to find the best hyperparameters
    clf = GridSearchCV(svm_clf, params, cv=skf, scoring='accuracy', n_jobs=-1)
    clf.fit(preprocessed, labels)
    clfs.append(clf)

# results to store each hyperparameter configuration, its mean accuracy and standard deviation
results = {}

for clf in clfs:
    cv_res = clf.cv_results_
    for i in range(len(cv_res['params'])):
        params = str(cv_res['params'][i])
        mean = cv_res['mean_test_score'][i]
        if params not in results:
            results[params] = []
        results[params].append(mean)

# iterate through the results and calculate the mean accuracy and standard deviation
for params in results:
    mean_acc = 0
    mean_std = 0
    for mean in results[params]:
        mean_acc += mean
    mean_acc /= len(results[params])
    for mean in results[params]:
        mean_std += (mean - mean_acc) ** 2

    mean_std = mean_std ** 0.5
    confidence_interval = 1.96 * (mean_std / (len(results[params]) ** 0.5))
    # print the results using last 2 digits after the decimal point
    print(f"Configuration: {params} | mean accuracy: {mean_acc:.2f} | Confidence interval: {mean_acc:.2f} +/- {confidence_interval:.3f}")



