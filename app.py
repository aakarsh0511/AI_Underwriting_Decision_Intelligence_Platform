import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import (
    load_dataset,
    get_dataset_summary,
    get_missing_value_summary,
    get_data_type_summary,
    get_high_cardinality_columns,
    get_constant_columns,
    get_target_distribution
)
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.risk_engine import RiskEngine

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
    "Dataset Overview",
    "Feature Engineering",
    "Preprocessing",
    "Model Training",
    "Risk Intelligence"
]
)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

@st.cache_data
def get_data():
    return load_dataset()


df = get_data()
engineer = FeatureEngineer(df)
engineered_df = engineer.engineer_features()
feature_info = engineer.get_feature_descriptions()
engineered_features = engineer.get_engineered_features()
preprocessor = DataPreprocessor(engineered_df)
manual_dropped = preprocessor.remove_unnecessary_columns()
high_missing = preprocessor.remove_high_missing_columns()
preprocessor.handle_missing_values()
preprocessor.prepare_target()
preprocessor.encode_categorical()
preprocessor.scale_features()
preprocessor.split_data()
X_train, X_test, y_train, y_test = (
    preprocessor.get_train_test_data())
processed_df = preprocessor.get_processed_data()
pre_summary = preprocessor.preprocessing_summary()
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

elif page == "Feature Engineering":

    st.header("🏦 Financial Feature Engineering")

    st.success("Business features generated successfully.")

    # -----------------------------------------

    st.subheader("Feature Description")

    description_df = pd.DataFrame(

        feature_info.items(),

        columns=[

            "Feature",

            "Business Meaning"

        ]

    )

    st.dataframe(
        description_df,
        use_container_width=True
    )

    # -----------------------------------------

    st.divider()

    st.subheader("Engineered Dataset")

    st.dataframe(

        engineered_df[engineered_features].head(),

        use_container_width=True

    )

    # -----------------------------------------

    st.divider()

    st.subheader("Summary Statistics")

    st.dataframe(

        engineered_df[engineered_features].describe(),
        use_container_width=True
    )
    # -----------------------------------------
    st.divider()
    st.subheader("Feature Distribution")
    feature = st.selectbox(
        "Select Feature",
        engineered_features
    )
    fig = px.histogram(
        engineered_df,
        x=feature,
        nbins=30,
        title=f"{feature} Distribution"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.divider()
    st.subheader("Outlier Detection")
    fig = px.box(
        engineered_df,
        y=feature,
        title=f"{feature} Box Plot"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.divider()
    st.subheader("Correlation Matrix")
    corr = engineered_df[engineered_features].corr(
        numeric_only=True
    )
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )
    # -----------------------------------------
    st.divider()
    st.subheader("Loan Status Distribution")
    if "loan_status" in engineered_df.columns:
        fig = px.histogram(
            engineered_df,
            x="loan_status",
            color="loan_status"
        )
        st.plotly_chart(
            fig,
            use_container_width=True )
    st.divider()
    st.download_button(
        label="📥 Download Engineered Dataset",
        data=engineered_df.to_csv(index=False),
        file_name="engineered_dataset.csv",
        mime="text/csv")

elif page == "Preprocessing":

    st.header("🧹 Data Preprocessing")

    st.success("Preprocessing Completed Successfully")

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        pre_summary["Rows"]
    )

    col2.metric(
        "Columns",
        pre_summary["Columns"]
    )

    col3.metric(
        "Remaining Missing Values",
        pre_summary["Missing Values"]
    )

    st.divider()

    st.subheader("Columns Removed Manually")

    st.write(manual_dropped)

    st.divider()

    st.subheader("Columns Removed (>90% Missing)")

    st.write(high_missing)

    st.divider()

    st.subheader("Processed Dataset Preview")

    st.dataframe(
        processed_df.head(),
        use_container_width=True
    )

    st.divider()

    st.subheader("Processed Dataset Shape")

    st.write(processed_df.shape)

    st.divider()

    st.subheader("Train Test Split")

    col1, col2 = st.columns(2)

    col1.metric(
        "Training Samples",X_train.shape[0])
    col2.metric("Testing Samples",X_test.shape[0])
    st.divider()
    st.subheader("Model Features")
    st.write(X_train.columns.tolist())
    st.divider()
    st.subheader("Target Distribution")
    st.write(y_train.value_counts())
    st.subheader("Target Class Distribution")
    st.dataframe(y_train.value_counts().rename_axis("Class").reset_index(name="Count"))

elif page == "Model Training":

    st.header("🤖 Model Training & Comparison")

    if st.button("Train Models"):

        trainer = ModelTrainer()

        results = trainer.train_models(
            X_train,
            y_train,
            X_test,
            y_test
        )

        best_model = trainer.get_best_model()
        trained_model = trainer.get_model(best_model)

        # Create Risk Engine
        risk_engine = RiskEngine(trained_model)

        # Generate complete risk report
        risk_report = risk_engine.generate_risk_report(X_test)

        # Save everything
        st.session_state["results"] = results
        st.session_state["best_model"] = best_model
        st.session_state["trained_model"] = trained_model
        st.session_state["risk_report"] = risk_report

    if "results" in st.session_state:

        st.success("Models trained successfully.")

        st.subheader("Model Performance")

        st.dataframe(
            st.session_state["results"],
            use_container_width=True
        )

        st.metric(
            "Best Model",
            st.session_state["best_model"]
        )


elif page == "Risk Intelligence":

    st.header("🛡️ Individual Loan Risk Assessment")

    # Check if models have been trained
    if "trained_model" not in st.session_state:
        st.warning("⚠️ Please train the model first.")
        st.stop()

    # Recreate Risk Engine
    trained_model = st.session_state["trained_model"]
    risk_engine = RiskEngine(trained_model)

    st.info("Select a loan application to assess.")

    customer_id = st.selectbox(
        "Select Loan Application",
        options=range(len(X_test))
    )

    customer_features = X_test.iloc[[customer_id]]

    prediction = risk_engine.assess_single_customer(
        customer_features
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Probability of Default",
        f"{prediction['Probability of Default']:.2f}%"
    )

    col2.metric(
        "Risk Score",
        f"{prediction['Risk Score']:.2f}"
    )

    col3.metric(
        "Confidence",
        f"{prediction['Confidence']:.2f}%"
    )

    st.divider()

    col1, col2 = st.columns(2)

    col1.metric(
        "Risk Bucket",
        prediction["Risk Bucket"]
    )

    col2.metric(
        "Recommendation",
        prediction["Recommendation"]
    )

    st.divider()

    st.subheader("Customer Features")

    feature_df = pd.DataFrame({
        "Feature": customer_features.columns,
        "Value": customer_features.iloc[0].values
    })

    st.dataframe(
        feature_df,
        use_container_width=True
    )

