import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class ModelTrainer:

    def __init__(self):

        self.models = {

            "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

            "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )

        }

        self.results = {}

    # ---------------------------------------------

    def train_models(
            self,
            X_train,
            y_train,
            X_test,
            y_test
    ):

        for name, model in self.models.items():

            model.fit(
                X_train,
                y_train
            )

            prediction = model.predict(
                X_test
            )

            probability = model.predict_proba(
                X_test
            )[:, 1]

            self.results[name] = {

                "Accuracy":
                round(
                    accuracy_score(
                        y_test,
                        prediction
                    ),
                    4
                ),

                "Precision":
                round(
                    precision_score(
                        y_test,
                        prediction
                    ),
                    4
                ),

                "Recall":
                round(
                    recall_score(
                        y_test,
                        prediction
                    ),
                    4
                ),

                "F1 Score":
                round(
                    f1_score(
                        y_test,
                        prediction
                    ),
                    4
                ),

                "ROC AUC":
                round(
                    roc_auc_score(
                        y_test,
                        probability
                    ),
                    4
                )

            }

        return pd.DataFrame(
            self.results
        ).T

    # ---------------------------------------------

    def get_best_model(self):

        best = max(
            self.results,
            key=lambda x: self.results[x]["ROC AUC"]
        )

        return best

    def get_model(self, model_name):

        return self.models[model_name]