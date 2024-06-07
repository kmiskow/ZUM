from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.covariance import EllipticEnvelope

def calc_metrics(y_true,y_pred):
    y_pred_binary = np.where(y_pred == 1, 1, -1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[1, -1]).ravel()
    precision = precision_score(y_true, y_pred_binary, pos_label=-1)
    recall = recall_score(y_true, y_pred_binary, pos_label=-1)
    f1 = f1_score(y_true, y_pred_binary, pos_label=-1)
    accuracy = accuracy_score(y_true, y_pred_binary)
    print(f'True Positives (TP): {tp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Accuracy: {accuracy:.2f}')

def isolation_forest(X,y_true,debug = False):
    clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    y_pred = clf.fit_predict(X)

    calc_metrics(y_true,y_pred)

    if debug:
        plt.figure(figsize=(10, 6))
        plt.title("Global Outlier Detection using Isolation Forest")
        plt.scatter(X[:, 0], X[:, 1], c='white', s=20, edgecolor='k')


        normal_mask = y_pred == 1
        plt.scatter(X[normal_mask, 0], X[normal_mask, 1], c='blue', s=20, edgecolor='k', label='Inliers')

        outlier_mask = y_pred == -1
        plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], c='red', s=20, edgecolor='k', label='Outliers')

        plt.legend()
        plt.show()


def one_class_svm(X, y_true, debug=False):
    clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    y_pred = clf.fit_predict(X)

    calc_metrics(y_true, y_pred)

    if debug:
        plt.figure(figsize=(10, 6))
        plt.title("Global Outlier Detection using One-Class SVM")
        plt.scatter(X[:, 0], X[:, 1], c='white', s=20, edgecolor='k')

        normal_mask = y_pred == 1
        plt.scatter(X[normal_mask, 0], X[normal_mask, 1], c='blue', s=20, edgecolor='k', label='Inliers')

        outlier_mask = y_pred == -1
        plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], c='red', s=20, edgecolor='k', label='Outliers')

        plt.legend()
        plt.show()

def sgd_one_class_svm(X, y_true, debug=False):
    clf = SGDOneClassSVM(nu=0.1, shuffle=True, random_state=42)
    y_pred = clf.fit_predict(X)

    calc_metrics(y_true, y_pred)

    if debug:
        plt.figure(figsize=(10, 6))
        plt.title("Global Outlier Detection using SGD One-Class SVM")
        plt.scatter(X[:, 0], X[:, 1], c='white', s=20, edgecolor='k')

        normal_mask = y_pred == 1
        plt.scatter(X[normal_mask, 0], X[normal_mask, 1], c='blue', s=20, edgecolor='k', label='Inliers')

        outlier_mask = y_pred == -1
        plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], c='red', s=20, edgecolor='k', label='Outliers')

        plt.legend()
        plt.show()

def elliptic_envelope(X, y_true, debug=False):
    clf = EllipticEnvelope(contamination=0.1, random_state=42)
    y_pred = clf.fit_predict(X)

    y_pred = np.where(y_pred == 1, 1, 0)

    calc_metrics(y_true, y_pred)

    if debug:
        plt.figure(figsize=(10, 6))
        plt.title("Global Outlier Detection using Elliptic Envelope")
        plt.scatter(X[:, 0], X[:, 1], c='white', s=20, edgecolor='k')

        normal_mask = y_pred == 1
        plt.scatter(X[normal_mask, 0], X[normal_mask, 1], c='blue', s=20, edgecolor='k', label='Inliers')

        outlier_mask = y_pred == 0
        plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], c='red', s=20, edgecolor='k', label='Outliers')

        plt.legend()
        plt.show()