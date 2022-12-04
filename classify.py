import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

def model_selection(x, y,number_of_cv_to_perform=5,metric_to_eval = 'f1'):
    """
     Arguments:
        x: input features
        y: labels
        number_of_cv_to_perform: number of k-folds
        metric_to_eval: evaluation score for selecting the best model
    Returns:
        the best performing model to be trained.
    """
    classfiers_names = [
        # "Nearest Neighbors",
        # "Linear SVM",
        "RBF-SVM",
        # "Gaussian Process",
        # "Decision Tree",
        "Random-Forest",
        "Neural-Net(minor regularization)",
        "Neural-Net(moderate regularization)",
        "Neural-Net(high regularization)",
        "AdaBoost",
        "Naive-Bayes",
        # "QDA",
        "XGB with (n_estimators=2, max_depth=2)",
        "XGB with (n_estimators=10, max_depth=10)",
        "XGB with (n_estimators=100, max_depth=100)",
    ]
    # TODO : hyperparameteres optimization using optuna
    classifiers_to_eval = [
        # KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        # DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=10, max_iter=1000),
        MLPClassifier(alpha=1, max_iter=1000),
        MLPClassifier(alpha=0.1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        # QuadraticDiscriminantAnalysis(),
        XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic'),
        XGBClassifier(n_estimators=10, max_depth=10, learning_rate=1, objective='binary:logistic'),
        XGBClassifier(n_estimators=100, max_depth=100, learning_rate=1, objective='binary:logistic'),
    ]

    classifiers_metrics = []
    for clf_name, clf in zip(classfiers_names, classifiers_to_eval):
        curr_metrics = cross_val_score(clf, x, y,  scoring=metric_to_eval, cv=number_of_cv_to_perform)
        classifiers_metrics.append(np.mean(curr_metrics))
        print('The evaluation of %s resulted with an average %s score of  %g'
              % (clf_name, metric_to_eval, classifiers_metrics[-1]))
    best_model = classifiers_to_eval[np.argmax(classifiers_metrics)]

    return best_model
