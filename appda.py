import streamlit as st
import pandas as pd
from DataPreprocessing import DataPreprocessor  # import your class here

st.title("Simple Data Preprocessor Frontend")

st.markdown("""
Upload a CSV dataset, and the tool will:
- Detect target column
- Show data sanity checks
- Show categorical distributions
- Provide EDA summary
- Handle missing values and scaling
- Recommend feature selection and modeling techniques
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data")
    st.dataframe(df.head())

    dp = DataPreprocessor()

    # Detect target column
    target_col = dp.detect_target_column(df)
    st.write(f"**Detected target column:** `{target_col}`")

    # Show sanity check
    st.write("### Data Sanity Check")
    if dp.sanity_check(df):
        st.write("Sanity check passed!")
    else:
        st.write("Sanity check failed.")

    # Show categorical distributions
    st.write("### Categorical Distributions")
    dp.check_categorical_distributions(df)

    # EDA summary
    st.write("### EDA Summary")
    dp.eda_summary(df)

    # Process the data
    processed_df, target_col = dp.preprocess(df)

    st.write("### Processed Data (first 5 rows)")
    st.dataframe(processed_df.head())

    st.write("### Target column")
    st.write(target_col)

    st.write("### Recommended Methods")
    recs = dp.recommend_methods(
        n_features=processed_df.shape[1],
        n_samples=processed_df.shape[0],
        has_target=True
    )
    st.write("**Feature Selection:**", recs['feature_selection'])
    st.write("**Modeling:**", recs['modeling'])
