from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def knn_classifier(features_train, features_test, target_train, target_test, n_neighbors=3):

    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)

    predictions = neigh.fit(features_train, target_train).predict(features_test)

    accuracy = accuracy_score(target_test, predictions)

    return neigh, accuracy


def naive_bayes_classifier(features_train, features_test, target_train, target_test):

    gnb = GaussianNB()

    predictions = gnb.fit(features_train, target_train).predict(features_test)

    accuracy = accuracy_score(target_test, predictions)

    return gnb, accuracy

def logistic_regression_classifier(features_train, features_test, target_train, target_test):

    logreg = LogisticRegression()

    predictions = logreg.fit(features_train, target_train).predict(features_test)

    accuracy = accuracy_score(target_test, predictions)

    return logreg, accuracy

def decision_tree_classifier(features_train, features_test, target_train, target_test):

    tree = DecisionTreeClassifier()

    predictions = tree.fit(features_train, target_train).predict(features_test)

    accuracy = accuracy_score(target_test, predictions)

    return tree, accuracy

def random_forest_classifier(features_train, features_test, target_train, target_test):

    rf = RandomForestClassifier(random_state=42)

    predictions = rf.fit(features_train, target_train).predict(features_test)

    accuracy = accuracy_score(target_test, predictions)

    return rf, accuracy