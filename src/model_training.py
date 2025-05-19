from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def knn_classifier(features_train, target_train, n_neighbors=3):

    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)

    neigh.fit(features_train, target_train)

    return neigh


def naive_bayes_classifier(features_train, target_train):

    gnb = GaussianNB()

    gnb.fit(features_train, target_train)

    return gnb

def logistic_regression_classifier(features_train, target_train):

    logreg = LogisticRegression()

    logreg.fit(features_train, target_train)

    return logreg

def decision_tree_classifier(features_train, target_train):

    tree = DecisionTreeClassifier()

    tree.fit(features_train, target_train)

    return tree

def random_forest_classifier(features_train, target_train):

    rf = RandomForestClassifier(random_state=42)

    rf.fit(features_train, target_train)

    return rf


def svm_classifier(features_train, target_train):

    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

    svm_model.fit(features_train, target_train)

    return svm_model