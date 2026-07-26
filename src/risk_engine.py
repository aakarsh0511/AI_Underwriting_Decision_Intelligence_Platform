import pandas as pd
import numpy as np


class RiskEngine:

    def __init__(self, model):
        self.model = model

    # -------------------------------------------------
    # Generate Full Risk Report
    # -------------------------------------------------

    def generate_risk_report(self, X):

        probability = self.model.predict_proba(X)[:, 1]

        report = pd.DataFrame()

        report["Probability of Default"] = probability

        # Higher = Riskier
        report["Risk Score"] = (probability * 100).round(2)

        report["Confidence"] = (
            np.maximum(probability, 1 - probability) * 100
        ).round(2)

        report["Risk Bucket"] = report[
            "Probability of Default"
        ].apply(self.get_risk_bucket)

        report["Recommendation"] = report[
            "Risk Bucket"
        ].apply(self.get_recommendation)

        return report

    # -------------------------------------------------

    def assess_single_customer(self, X_row):

        probability = self.model.predict_proba(X_row)[0][1]

        return {

            "Probability of Default":
                round(probability * 100, 2),

            "Risk Score":
                round(probability * 100, 2),

            "Confidence":
                round(
                    max(probability, 1 - probability) * 100,
                    2
                ),

            "Risk Bucket":
                self.get_risk_bucket(probability),

            "Recommendation":
                self.get_recommendation(
                    self.get_risk_bucket(probability)
                )

        }

    # -------------------------------------------------

    def get_risk_bucket(self, pd_value):

        if pd_value < 0.05:
            return "🟢 Very Low"

        elif pd_value < 0.15:
            return "🟢 Low"

        elif pd_value < 0.30:
            return "🟡 Medium"

        elif pd_value < 0.50:
            return "🟠 High"

        else:
            return "🔴 Very High"

    # -------------------------------------------------

    def get_recommendation(self, bucket):

        mapping = {

            "🟢 Very Low": "Approve",

            "🟢 Low": "Approve",

            "🟡 Medium": "Manual Review",

            "🟠 High": "Senior Review",

            "🔴 Very High": "Reject"

        }

        return mapping[bucket]