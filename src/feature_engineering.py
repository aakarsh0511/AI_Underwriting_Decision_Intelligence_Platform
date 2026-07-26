import pandas as pd
import numpy as np

class FeatureEngineer:

    def __init__(self, df):
        self.df = df.copy()

    def engineer_features(self):

        # -----------------------------
        # Loan Term
        # -----------------------------

        if "term" in self.df.columns:

            self.df["term_months"] = (
                self.df["term"]
                .astype(str)
                .str.extract(r'(\d+)')
                .astype(float)
            )

        # -----------------------------
        # Interest Rate
        # -----------------------------

        if "int_rate" in self.df.columns:

            self.df["interest_rate"] = (
                self.df["int_rate"]
                .astype(str)
                .str.replace("%", "", regex=False)
                .astype(float)
            )

        # -----------------------------
        # Monthly Income
        # -----------------------------

        if "annual_inc" in self.df.columns:

            self.df["monthly_income"] = (
                self.df["annual_inc"] / 12
            )

        # -----------------------------
        # Loan to Income Ratio
        # -----------------------------

        if {"loan_amnt", "annual_inc"}.issubset(self.df.columns):

            self.df["loan_to_income_ratio"] = np.where(
                self.df["annual_inc"] > 0,
                self.df["loan_amnt"] / self.df["annual_inc"],
                np.nan
            )

        # -----------------------------
        # EMI Burden
        # -----------------------------

        if {"installment", "monthly_income"}.issubset(self.df.columns):
            self.df["emi_burden"] = np.where(
            self.df["monthly_income"] > 0,
            self.df["installment"] / self.df["monthly_income"],
            np.nan
        )

        # -----------------------------
        # Average FICO
        # -----------------------------

        if {"fico_range_low", "fico_range_high"}.issubset(self.df.columns):

            self.df["avg_fico"] = (
                self.df["fico_range_low"] +
                self.df["fico_range_high"]
            ) / 2

        # -----------------------------
        # Revolving Utilization
        # -----------------------------

        if "revol_util" in self.df.columns:

            self.df["revol_util_clean"] = (
                self.df["revol_util"]
                .astype(str)
                .str.replace("%", "", regex=False)
            )

            self.df["revol_util_clean"] = pd.to_numeric(
                self.df["revol_util_clean"],
                errors="coerce"
            )

        # -----------------------------
        # Employment Length
        # -----------------------------

        if "emp_length" in self.df.columns:

            self.df["employment_years"] = (
                self.df["emp_length"]
                .astype(str)
                .str.extract(r'(\d+)')
            )

            self.df["employment_years"] = pd.to_numeric(
                self.df["employment_years"],
                errors="coerce"
            ).fillna(0)

        return self.df

        # --------------------------------------------------
    # Feature Descriptions
    # --------------------------------------------------

    def get_feature_descriptions(self):

        descriptions = {

            "term_months":
            "Loan tenure in months.",

            "interest_rate":
            "Annual interest rate charged on the loan.",

            "monthly_income":
            "Monthly income derived from annual income.",

            "loan_to_income_ratio":
            "Loan Amount divided by Annual Income.",

            "emi_burden":
            "Monthly EMI divided by Monthly Income.",

            "avg_fico":
            "Average of FICO Low and High score.",

            "revol_util_clean":
            "Credit Card Revolving Utilization (%).",

            "employment_years":
            "Years of employment converted into numeric."

        }

        return descriptions


    # --------------------------------------------------
    # Correlation
    # --------------------------------------------------

    def get_engineered_features(self):

        return [

            "term_months",

            "interest_rate",

            "monthly_income",

            "loan_to_income_ratio",

            "emi_burden",

            "avg_fico",

            "revol_util_clean",

            "employment_years"

        ]
    