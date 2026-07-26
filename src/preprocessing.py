import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:

    def __init__(self, df):
        self.df = df.copy()

        # Columns that should never go into the ML model
        self.columns_to_drop = [
            "id",
            "member_id",
            "url",
            "desc",
            "emp_title",
            "title",
            "zip_code",
            "issue_d",
            "earliest_cr_line",
            "last_pymnt_d",
            "next_pymnt_d",
            "last_credit_pull_d"
        ]

    # ----------------------------------------------------
    # Drop unnecessary columns
    # ----------------------------------------------------

    def remove_unnecessary_columns(self):

        dropped = []

        for col in self.columns_to_drop:

            if col in self.df.columns:

                self.df.drop(columns=col, inplace=True)

                dropped.append(col)

        return dropped

    # ----------------------------------------------------
    # Remove columns having too many missing values
    # ----------------------------------------------------

    def remove_high_missing_columns(
            self,
            threshold=90
    ):

        missing_percent = (
            self.df.isnull().mean() * 100
        )

        cols = missing_percent[
            missing_percent > threshold
        ].index.tolist()

        self.df.drop(
            columns=cols,
            inplace=True
        )

        return cols

    # ----------------------------------------------------
    # Fill missing values
    # ----------------------------------------------------

    def handle_missing_values(self):

        numeric_cols = self.df.select_dtypes(
            include=np.number
        ).columns

        categorical_cols = self.df.select_dtypes(
            include="object"
        ).columns

        self.df[numeric_cols] = (
            self.df[numeric_cols]
            .fillna(
                self.df[numeric_cols].median()
            )
        )

        self.df[categorical_cols] = (
            self.df[categorical_cols]
            .fillna("Unknown")
        )

    # ----------------------------------------------------
    # Dataset summary
    # ----------------------------------------------------

    def preprocessing_summary(self):

        summary = {

            "Rows": self.df.shape[0],

            "Columns": self.df.shape[1],

            "Missing Values":
            int(
                self.df
                .isnull()
                .sum()
                .sum()
            )

        }
        return summary
    # ----------------------------------------------------
    def get_processed_data(self):

        return self.df
        # ----------------------------------------------------
    # Encode Categorical Variables
    # ----------------------------------------------------

    def encode_categorical(self):

        self.label_encoders = {}
        categorical = self.df.select_dtypes(include="object").columns.tolist()

        if "loan_status" in categorical:
            categorical.remove("loan_status")

        for col in categorical:

            encoder = LabelEncoder()

            self.df[col] = encoder.fit_transform(
                self.df[col].astype(str)
            )

            self.label_encoders[col] = encoder

    # ----------------------------------------------------
    # Scale Numeric Features
    # ----------------------------------------------------

    def scale_features(self, target="loan_status"):

        # Replace infinity values
        self.df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        numeric_cols = self.df.select_dtypes(
            include="number"
        ).columns.tolist()

        if target in numeric_cols:
            numeric_cols.remove(target)

        # Fill NaN created after replacing infinity
        self.df[numeric_cols] = (
            self.df[numeric_cols]
            .fillna(
                self.df[numeric_cols].median()
            )
        )

        self.scaler = StandardScaler()

        self.df[numeric_cols] = self.scaler.fit_transform(
            self.df[numeric_cols]
        )

    # ----------------------------------------------------
    # Train Test Split
    # ----------------------------------------------------

    def split_data(
            self,
            target="loan_status",
            test_size=0.2,
            random_state=42
    ):
        X = self.df.drop(columns=[target])
        y = self.df[target]
        if y.value_counts().min() >= 2:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state)
        

    # ----------------------------------------------------
    # Save Pipeline
    # ----------------------------------------------------

    def save_pipeline(self):

        joblib.dump(
            self.scaler,
            "models/scaler.pkl"
        )

        joblib.dump(
            self.label_encoders,
            "models/label_encoders.pkl"
        )

    # ----------------------------------------------------

    def get_train_test_data(self):

        return (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        )
    def remove_infinite_values(self):
        self.df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(
            self.df[numeric_cols].median()
        )
    # ----------------------------------------------------
# Prepare Target Variable
# ----------------------------------------------------

    def prepare_target(self):

        if "loan_status" not in self.df.columns:
            return

        mapping = {
            "Fully Paid": 0,
            "Charged Off": 1
        }

        # Keep only the two classes
        self.df = self.df[
            self.df["loan_status"].isin(mapping.keys())
        ].copy()

        self.df["loan_status"] = (
            self.df["loan_status"]
            .map(mapping)
        )