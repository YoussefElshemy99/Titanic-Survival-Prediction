import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def knn_classifier(features_train, features_test, target_train, target_test, n_neighbors=3):

    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)

    neigh.fit(features_train, target_train)

    predictions = neigh.predict(features_test)

    #  print(predictions)

    accuracy = accuracy_score(target_test, predictions)

    return accuracy


def naive_bayes_classifier(features_train, features_test, target_train, target_test):

    gnb = GaussianNB()

    predictions = gnb.fit(features_train, target_train).predict(features_test)

    #  print(predictions)

    accuracy = accuracy_score(target_test, predictions)

    return accuracy

def logistic_regression_classifier(features_train, features_test, target_train, target_test):

    from sklearn.linear_model import LogisticRegression

    logreg = LogisticRegression()

    predictions = logreg.fit(features_train, target_train).predict(features_test)

    accuracy = accuracy_score(target_test, predictions)

    return accuracy

def decision_tree_classifier(features_train, features_test, target_train, target_test):

    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier()

    predictions = tree.fit(features_train, target_train).predict(features_test)

    accuracy = accuracy_score(target_test, predictions)

    return accuracy