import pandas as pd
import numpy as np

from config import DATASET_PATH


def load_dataset():
    """
    Load Lending Club Dataset
    """

    try:

        df = pd.read_excel(DATASET_PATH)

        return df

    except Exception as e:

        raise Exception(f"Error Loading Dataset : {e}")


# =====================================================
# DATASET SUMMARY
# =====================================================

def get_dataset_summary(df):

    summary = {

        "Total Rows": df.shape[0],

        "Total Columns": df.shape[1],

        "Numeric Columns":
        len(df.select_dtypes(include="number").columns),

        "Categorical Columns":
        len(df.select_dtypes(include="object").columns),

        "Duplicate Records":
        int(df.duplicated().sum()),

        "Memory Usage (MB)":
        round(df.memory_usage(deep=True).sum() / (1024**2), 2)

    }

    return summary


# =====================================================
# MISSING VALUES
# =====================================================

def get_missing_value_summary(df):

    missing = pd.DataFrame({

        "Column": df.columns,

        "Missing Values": df.isnull().sum().values,

        "Missing %":
        (df.isnull().mean() * 100).round(2)

    })

    missing = missing.sort_values(

        by="Missing %",
        ascending=False

    )

    return missing


# =====================================================
# DATA TYPES
# =====================================================

def get_data_type_summary(df):

    datatype = (

        df.dtypes
        .astype(str)
        .value_counts()
        .reset_index()

    )

    datatype.columns = [

        "Data Type",
        "Count"

    ]

    return datatype


# =====================================================
# HIGH CARDINALITY
# =====================================================

def get_high_cardinality_columns(df,
                                 threshold=50):

    records = []

    for col in df.select_dtypes(include="object"):

        unique = df[col].nunique()

        if unique > threshold:

            records.append({

                "Column": col,

                "Unique Values": unique

            })

    return pd.DataFrame(records)


# =====================================================
# CONSTANT COLUMNS
# =====================================================

def get_constant_columns(df):

    constant = []

    for col in df.columns:

        if df[col].nunique(dropna=False) == 1:

            constant.append(col)

    return constant


# =====================================================
# TARGET DISTRIBUTION
# =====================================================

def get_target_distribution(df,
                            target="loan_status"):

    if target not in df.columns:

        return pd.DataFrame()

    target_df = (

        df[target]
        .value_counts()
        .reset_index()

    )

    target_df.columns = [

        target,
        "Count"

    ]

    return target_df