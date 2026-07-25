import streamlit as st
import pandas as pd

from src.data_loader import (
    load_dataset,
    get_dataset_summary,
    get_missing_value_summary,
    get_data_type_summary,
    get_high_cardinality_columns,
    get_constant_columns,
    get_target_distribution
)

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Underwriting Decision Intelligence Platform",
    page_icon="🏦",
    layout="wide"
)

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("🏦 AI Underwriting Decision Intelligence Platform")

st.markdown("""
This platform assists underwriters in making intelligent lending decisions using

- Machine Learning
- Explainable AI
- Policy Engine
- Loan Optimization
- Scenario Simulation
- AI Copilot
""")

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Dataset Overview"
    ]
)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

@st.cache_data
def get_data():
    return load_dataset()


df = get_data()

summary = get_dataset_summary(df)
missing = get_missing_value_summary(df)
datatype = get_data_type_summary(df)
high_cardinality = get_high_cardinality_columns(df)
constant_columns = get_constant_columns(df)
target_distribution = get_target_distribution(df)

# ---------------------------------------------------
# HOME
# ---------------------------------------------------

if page == "Home":

    st.header("Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Rows",
        f"{summary['Total Rows']:,}"
    )

    col2.metric(
        "Total Columns",
        summary["Total Columns"]
    )

    col3.metric(
        "Memory Usage (MB)",
        summary["Memory Usage (MB)"]
    )

    st.divider()

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Numeric Columns",
        summary["Numeric Columns"]
    )

    col5.metric(
        "Categorical Columns",
        summary["Categorical Columns"]
    )

    col6.metric(
        "Duplicate Records",
        summary["Duplicate Records"]
    )

    st.divider()

    st.subheader("Project Workflow")

    st.markdown("""
Customer Application

↓

Data Validation

↓

Feature Engineering

↓

Credit Risk Prediction

↓

Risk Intelligence

↓

Underwriting Policy Engine

↓

Explainable AI

↓

Loan Optimization

↓

Scenario Simulation

↓

Counterfactual Recommendation

↓

Historical Case Intelligence

↓

AI Underwriting Copilot

↓

Final Lending Proposal

↓

Audit Trail & Executive Report
""")

# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------

elif page == "Dataset Overview":

    st.header("Dataset Overview")

    # -----------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # -----------------------------------------

    st.divider()

    st.subheader("Dataset Shape")

    st.write(f"Rows : {df.shape[0]:,}")

    st.write(f"Columns : {df.shape[1]}")

    # -----------------------------------------

    st.divider()

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

    # -----------------------------------------

    st.divider()

    st.subheader("Missing Value Summary")

    st.dataframe(missing)

    # -----------------------------------------

    st.divider()

    st.subheader("Data Type Summary")

    st.dataframe(datatype)

    # -----------------------------------------

    st.divider()

    st.subheader("High Cardinality Columns")

    if len(high_cardinality) == 0:

        st.success("No High Cardinality Columns Found.")

    else:

        st.dataframe(high_cardinality)

    # -----------------------------------------

    st.divider()

    st.subheader("Constant Columns")

    if len(constant_columns) == 0:

        st.success("No Constant Columns Found.")

    else:

        st.write(constant_columns)

    # -----------------------------------------

    st.divider()

    st.subheader("Loan Status Distribution")

    if not target_distribution.empty:

        st.dataframe(target_distribution)

    else:

        st.warning("Target column not found.")