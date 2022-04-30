from hdc import VectorClassifier

class HDTrainer:
    def __init__(self, n_models):
        self.n_models = n_models

    def train(self, train_X, train_Y, test_X, test_Y, n_classes, optimize=None, log_path=None):

        agg_clf = VectorClassifier(n_classes=n_classes, dim=train_X.shape[1])
        agg_clf.construct_memory(train_X, train_Y)
        if optimize is None:
            _, _, train_acc = agg_clf.predict(train_X, train_Y) 
            _, _, test_acc = agg_clf.predict(test_X, test_Y)
            return agg_clf, train_acc, test_acc
            
        elif optimize == 'ErrorWeighted':
            tmp_train_X, tmp_train_Y = train_X.copy(), train_Y.copy()
            for _ in range(100):
                _, wrong_idx, cur_acc = agg_clf.predict(train_X, train_Y)
                tmp_train_X, tmp_train_Y = train_X[wrong_idx, :], train_Y[wrong_idx]
                if len(tmp_train_Y) == 0:
                    break
                clf = VectorClassifier(n_classes=n_classes, dim=train_X.shape[1])
                clf.construct_memory(tmp_train_X, tmp_train_Y)
                _, _, acc = clf.predict(tmp_train_X, tmp_train_Y)
                agg_clf.get_model(clf, w=acc * (1 - cur_acc))
                
                
        elif optimize == 'adaptHD':
            best_test_acc = 0
            agg_clf.construct_memory(train_X, train_Y)
            for _ in range(10):
                yhat, wrong_idx, cur_acc = agg_clf.predict(train_X, train_Y)
                agg_clf.optimize_label(train_X[wrong_idx, :], train_Y[wrong_idx], yhat[wrong_idx])
                _, _, acc = agg_clf.predict(test_X, test_Y)
                if acc > best_test_acc:
                    best_label = agg_clf.label.copy()
                    best_count = agg_clf.count.copy()
                    best_test_acc = acc
            agg_clf.label = best_label
            agg_clf.count = best_count
    
        _, _, train_acc = agg_clf.predict(train_X, train_Y) 
        _, _, test_acc = agg_clf.predict(test_X, test_Y)

        return agg_clf, train_acc, test_acc

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def classical_classification(train_real_X, train_Y, test_real_X, test_Y):
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        # "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    for name, clf in zip(names, classifiers):
        clf.fit(train_real_X, train_Y)
        score = clf.score(test_real_X, test_Y)
        print(f"{name}: {round(score * 100, 1)}%.")
