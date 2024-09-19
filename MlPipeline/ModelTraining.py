from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import mlflow
import streamlit as st


class ModelTraining:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def train_models(self):
        seed = 7
        models = []
        models.append(('LR', LogisticRegression(max_iter=200)))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(max_iter=500)))

        results = []
        names = []
        scoring = 'accuracy'

        with mlflow.start_run():
            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
                cv_results = model_selection.cross_val_score(model, self.X, self.Y, cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(name)

                # Log metrics to DagsHub (via MLflow)
                mlflow.log_metric(f"{name}_mean_accuracy", cv_results.mean())
                mlflow.log_metric(f"{name}_std_accuracy", cv_results.std())
                
                msg = f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})"
                st.write(msg)

        return results, names