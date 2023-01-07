import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from DataLoader import DataLoader


data_path = "../data/credit.data"

dataset, labels = DataLoader.load_credit_with_onehot(data_path)

# preprocess the data by using StandardScaler
scaler = StandardScaler()
preprocessed = scaler.fit_transform(dataset)

models = [KNeighborsClassifier(), SVC(), DecisionTreeClassifier(), RandomForestClassifier()]
params = [{'n_neighbors': [9, 29], 'weights': ['uniform']},
          {'kernel': ['linear'], 'C': [5, 500]},
          {'criterion': ['entropy'], 'max_depth': [2, 5]},
          {'n_estimators': [100], 'criterion': ['gini', 'entropy']}]

outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

normalized = MinMaxScaler(feature_range=(-1, 1)).fit_transform(dataset)

knn_results = []
knn_best_params = []
knn_f1 = []

svm_results = []
svm_best_params = []
svm_f1 = []

dt_results = []
dt_best_params = []
dt_f1 = []

rf_results = []
rf_best_params = []
rf_f1 = []

knn_results_index = 0
svm_results_index = 0
dt_results_index = 0
rf_results_index = 0


for i, (train, test) in enumerate(outer_cv.split(normalized, labels)):  # outer loop
    for model, param in zip(models, params):  # inner loop
        if isinstance(model, KNeighborsClassifier):  # KNN
            inner_clf = GridSearchCV(model, param, cv=inner_cv, scoring='accuracy', n_jobs=-1)
            inner_clf.fit(normalized[train], labels[train])

            knn_results.append([])
            knn_results[knn_results_index].append(inner_clf.cv_results_['mean_test_score'])
            knn_results_index += 1

            # run the model with the best parameters on the test set
            best_knn = KNeighborsClassifier(n_neighbors=inner_clf.best_params_['n_neighbors'],
                                            weights=inner_clf.best_params_['weights'])
            best_knn.fit(normalized[train], labels[train])
            knn_best_params.append(best_knn.score(normalized[test], labels[test]))
            # calculate F1 score
            knn_f1.append(f1_score(labels[test], best_knn.predict(normalized[test])))

        elif isinstance(model, SVC):  # SVM
            inner_clf = GridSearchCV(model, param, cv=inner_cv, scoring='accuracy', n_jobs=-1)
            inner_clf.fit(normalized[train], labels[train])

            svm_results.append([])
            svm_results[svm_results_index].append(inner_clf.cv_results_['mean_test_score'])
            svm_results_index += 1

            # run the model with the best parameters on the test set
            best_svm = SVC(kernel=inner_clf.best_params_['kernel'], C=inner_clf.best_params_['C'])
            best_svm.fit(normalized[train], labels[train])
            svm_best_params.append(best_svm.score(normalized[test], labels[test]))
            # calculate F1 score
            svm_f1.append(f1_score(labels[test], best_svm.predict(normalized[test])))

        elif isinstance(model, DecisionTreeClassifier):  # Decision Tree
            inner_clf = GridSearchCV(model, param, cv=inner_cv, scoring='accuracy', n_jobs=-1)
            inner_clf.fit(normalized[train], labels[train])

            dt_results.append([])
            dt_results[dt_results_index].append(inner_clf.cv_results_['mean_test_score'])
            dt_results_index += 1

            # run the model with the best parameters on the test set
            best_dt = DecisionTreeClassifier(criterion=inner_clf.best_params_['criterion'],
                                             max_depth=inner_clf.best_params_['max_depth'])
            best_dt.fit(normalized[train], labels[train])
            dt_best_params.append(best_dt.score(normalized[test], labels[test]))
            # calculate F1 score
            dt_f1.append(f1_score(labels[test], best_dt.predict(normalized[test])))

        elif isinstance(model, RandomForestClassifier):  # Random Forest
            for _ in range(5):
                inner_clf = GridSearchCV(model, param, cv=inner_cv, scoring='accuracy', n_jobs=-1)
                inner_clf.fit(normalized[train], labels[train])

                rf_results.append([])
                rf_results[rf_results_index].append(inner_clf.cv_results_['mean_test_score'])
                rf_results_index += 1

                # run the model with the best parameters on the test set
                best_rf = RandomForestClassifier(n_estimators=inner_clf.best_params_['n_estimators'],
                                                 criterion=inner_clf.best_params_['criterion'])
                best_rf.fit(normalized[train], labels[train])
                rf_best_params.append(best_rf.score(normalized[test], labels[test]))
                # calculate F1 score
                rf_f1.append(f1_score(labels[test], best_rf.predict(normalized[test])))


knn_mean = np.mean(knn_results, axis=0)
knn_std = np.std(knn_results, axis=0)

svm_mean = np.mean(svm_results, axis=0)
svm_std = np.std(svm_results, axis=0)

dt_mean = np.mean(dt_results, axis=0)
dt_std = np.std(dt_results, axis=0)

rf_mean = np.mean(rf_results, axis=0)
rf_std = np.std(rf_results, axis=0)


for model in models:
    if isinstance(model, KNeighborsClassifier):
        print('-----KNN-----')

        print('Configuration 1: N=9, weights = uniform')
        confidence_interval = 1.96 * (knn_std[0, 0] / np.sqrt(len(knn_results)))
        print(f'Mean Accuracy: {knn_mean[0, 0]:.3f}, Confidence Interval: , {knn_mean[0, 0]:.3f} +/- {confidence_interval:.3f}\n')

        print('Configuration 2: N=29, weights = uniform')
        confidence_interval = 1.96 * (knn_std[0, 1] / np.sqrt(len(knn_results)))
        print(f'Mean Accuracy: {knn_mean[0, 1]:.3f}, Confidence Interval: , {knn_mean[0, 1]:.3f} +/- {confidence_interval:.3f}\n')

        best_mean = np.mean(knn_best_params)
        best_std = np.std(knn_best_params)
        confidence_interval = 1.96 * (best_std / np.sqrt(len(knn_best_params)))
        print(f'Overall KNN Accuracy: {best_mean:.3f}, Accuracy Confidence Interval: {best_mean:.3f} +/- {confidence_interval:.3f}, '
              f'F1 Score: {np.mean(knn_f1):.3f}, F1 Score Confidence Interval: {np.mean(knn_f1):.3f} +/- {1.96 * (np.std(knn_f1) / np.sqrt(len(knn_f1))):.3f}\n')

    elif isinstance(model, SVC):
        print('-----SVM-----')

        print('Configuration 1: kernel = linear, C = 5')
        confidence_interval = 1.96 * (svm_std[0, 0] / np.sqrt(len(svm_results)))
        print(f'Mean Accuracy: {svm_mean[0, 0]:.3f}, Confidence Interval: , {svm_mean[0, 0]:.3f} +/- {confidence_interval:.3f}\n')

        print('Configuration 2: kernel = linear, C = 500')
        confidence_interval = 1.96 * (svm_std[0, 1] / np.sqrt(len(svm_results)))
        print(f'Mean Accuracy: {svm_mean[0, 1]:.3f}, Confidence Interval: , {svm_mean[0, 1]:.3f} +/- {confidence_interval:.3f}\n')

        best_mean = np.mean(svm_best_params)
        best_std = np.std(svm_best_params)
        confidence_interval = 1.96 * (best_std / np.sqrt(len(svm_best_params)))
        print(f'Overall SVM Accuracy: {best_mean:.3f}, Accuracy Confidence Interval: {best_mean:.3f} +/- {confidence_interval:.3f}, '
              f'F1 Score: {np.mean(svm_f1):.3f}, F1 Score Confidence Interval: {np.mean(svm_f1):.3f} +/- {1.96 * (np.std(svm_f1) / np.sqrt(len(svm_f1))):.3f}\n')

    elif isinstance(model, DecisionTreeClassifier):
        print('-----Decision Tree-----')

        print('Configuration 1: criterion = entropy, max_depth = 2')
        confidence_interval = 1.96 * (dt_std[0, 0] / np.sqrt(len(dt_results)))
        print(f'Mean Accuracy: {dt_mean[0, 0]:.3f}, Confidence Interval: , {dt_mean[0, 0]:.3f} +/- {confidence_interval:.3f}\n')

        print('Configuration 2: criterion = entropy, max_depth = 5')
        confidence_interval = 1.96 * (dt_std[0, 1] / np.sqrt(len(dt_results)))
        print(f'Mean Accuracy: {dt_mean[0, 1]:.3f}, Confidence Interval: , {dt_mean[0, 1]:.3f} +/- {confidence_interval:.3f}\n')

        best_mean = np.mean(dt_best_params)
        best_std = np.std(dt_best_params)
        confidence_interval = 1.96 * (best_std / np.sqrt(len(dt_best_params)))
        print(f'Overall Decision Tree Accuracy: {best_mean:.3f}, Accuracy Confidence Interval: {best_mean:.3f} +/- {confidence_interval:.3f}, '
              f'F1 Score: {np.mean(dt_f1):.3f}, F1 Score Confidence Interval: {np.mean(dt_f1):.3f} +/- {1.96 * (np.std(dt_f1) / np.sqrt(len(dt_f1))):.3f}\n')

    elif isinstance(model, RandomForestClassifier):
        print('-----Random Forest-----')

        print('Configuration 1: n_estimators = 100, criterion = gini')
        confidence_interval = 1.96 * (rf_std[0, 0] / np.sqrt(len(rf_results)))
        print(f'Mean Accuracy: {rf_mean[0, 0]:.3f}, Confidence Interval: , {rf_mean[0, 0]:.3f} +/- {confidence_interval:.3f}\n')

        print('Configuration 2: n_estimators = 100, criterion = entropy')
        confidence_interval = 1.96 * (rf_std[0, 1] / np.sqrt(len(rf_results)))
        print(f'Mean Accuracy: {rf_mean[0, 1]:.3f}, Confidence Interval: , {rf_mean[0, 1]:.3f} +/- {confidence_interval:.3f}\n')

        best_mean = np.mean(rf_best_params)
        best_std = np.std(rf_best_params)
        confidence_interval = 1.96 * (best_std / np.sqrt(len(rf_best_params)))
        print(f'Overall Random Forest Accuracy: {best_mean:.3f}, Accuracy Confidence Interval: {best_mean:.3f} +/- {confidence_interval:.3f}, '
              f'F1 Score: {np.mean(rf_f1):.3f}, F1 Score Confidence Interval: {np.mean(rf_f1):.3f} +/- {1.96 * (np.std(rf_f1) / np.sqrt(len(rf_f1))):.3f}\n')
