import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Your DataPreprocessor class (imported or copied here)
class DataPreprocessor:
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []
        
    def recommend_methods(self, n_features, n_samples, has_target):
        recommendations = {
            "feature_selection": [],
            "modeling": []
        }
        
        if not has_target:
            recommendations["feature_selection"].append("Variance Threshold")
            recommendations["feature_selection"].append("Unsupervised clustering-based selection")
            recommendations["modeling"].append("KMeans")
            recommendations["modeling"].append("PCA")
            recommendations["modeling"].append("Autoencoders")
        else:
            if n_features > 100 or n_samples > 5000:
                recommendations["feature_selection"].append("Variance Threshold")
                recommendations["feature_selection"].append("Generic Univariate Feature Selection (e.g., SelectKBest with chi2, ANOVA)")
                recommendations["feature_selection"].append("Mutual Information")
                recommendations["modeling"].append("Linear / Logistic Regression")
                recommendations["modeling"].append("Ridge Regression")
                recommendations["modeling"].append("Random Forest (for embedded feature importance)")
                recommendations["modeling"].append("XGBoost / LightGBM")
            else:
                recommendations["feature_selection"].append("Recursive Feature Elimination (RFE) with SVM or Random Forest")
                recommendations["feature_selection"].append("Sequential Feature Selector (SFS)")
                recommendations["feature_selection"].append("Generic Univariate Feature Selection")
                recommendations["modeling"].append("SVM (SVC / SVR)")
                recommendations["modeling"].append("K-Nearest Neighbors (KNN)")
                recommendations["modeling"].append("Linear / Logistic Regression")
                recommendations["modeling"].append("Ridge Regression")
                recommendations["modeling"].append("Random Forest")
                recommendations["modeling"].append("Gradient Boosting Machines (XGBoost, LightGBM)")
        
        return recommendations

    def detect_target_column(self, df):
        """Auto-detect target column based on heuristics."""
        if df is None or df.empty:
            return None
            
        common_targets = ['target', 'label', 'class', 'species', 'outcome', 'y']
        
        for col in df.columns:
            if col.lower() in common_targets:
                return col
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_cols) == 1:
            return categorical_cols[0]
        
        if len(categorical_cols) > 1:
            unique_counts = df[categorical_cols].nunique()
            target_candidate = unique_counts.idxmin()
            return target_candidate
        
        return df.columns[-1]

    def handle_missing_values(self, df):
        if df.isnull().sum().sum() == 0:
            return df
            
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        df_processed = df.copy()
        
        if self.numerical_cols:
            num_imputer = SimpleImputer(strategy='median')
            df_processed[self.numerical_cols] = num_imputer.fit_transform(df[self.numerical_cols])
        
        if self.categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_processed[self.categorical_cols] = cat_imputer.fit_transform(df[self.categorical_cols])
        
        return df_processed

    def detect_outliers(self, df):
        outlier_summary = []
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_percentage = len(outliers) / len(df) * 100
            if outlier_percentage > 1:
                outlier_summary.append({
                    'Column': col,
                    'Outlier Count': len(outliers),
                    'Outlier %': round(outlier_percentage, 2)
                })
        
        return outlier_summary

    def choose_scaler(self, df):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            return None, "No numerical columns"
        
        outlier_cols = []
        sparse_cols = []
        
        for col in numerical_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(df) * 0.01:
                outlier_cols.append(col)
            
            zero_ratio = (df[col] == 0).sum() / len(df)
            if zero_ratio > 0.5:
                sparse_cols.append(col)
        
        if len(outlier_cols) > 0:
            return RobustScaler(), f"RobustScaler (outliers in {len(outlier_cols)} columns)"
        elif len(sparse_cols) > 0:
            return MaxAbsScaler(), f"MaxAbsScaler (sparse data in {len(sparse_cols)} columns)"
        else:
            return StandardScaler(), "StandardScaler (normal distribution assumed)"

    def apply_scaling(self, df, scaler):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df_scaled = df.copy()
        if len(numerical_cols) > 0 and scaler is not None:
            df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df_scaled

    def encode_categorical(self, df):
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        encoding_info = []
        
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            if unique_vals <= 2:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col])
                encoding_info.append(f"Label encoded '{col}' (binary)")
            elif unique_vals > 20:
                encoding_info.append(f"Warning: '{col}' has {unique_vals} unique values (high cardinality)")
                ohe = OneHotEncoder(sparse_output=False, drop='first')
                transformed = ohe.fit_transform(df[[col]])
                new_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                df_ohe = pd.DataFrame(transformed, columns=new_cols, index=df.index)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), df_ohe], axis=1)
            else:
                ohe = OneHotEncoder(sparse_output=False, drop='first')
                transformed = ohe.fit_transform(df[[col]])
                new_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                df_ohe = pd.DataFrame(transformed, columns=new_cols, index=df.index)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), df_ohe], axis=1)
                encoding_info.append(f"One-hot encoded '{col}' ({unique_vals} categories)")
        
        return df_encoded, encoding_info

# Streamlit App
def main():
    st.set_page_config(
        page_title="AutoML Data Preprocessor",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 AutoML Data Preprocessing Pipeline")
    st.markdown("Upload your dataset and get intelligent preprocessing recommendations!")

    # Initialize session state
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = DataPreprocessor()
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your dataset in CSV format"
    )

    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            
            st.sidebar.success(f"✅ File loaded successfully!")
            st.sidebar.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            return

        # Target column selection
        st.sidebar.subheader("🎯 Target Column")
        auto_target = st.session_state.preprocessor.detect_target_column(df)
        
        target_options = ['Auto-detect'] + list(df.columns)
        target_choice = st.sidebar.selectbox(
            "Select target column:",
            target_options,
            help=f"Auto-detected: {auto_target}"
        )
        
        if target_choice == 'Auto-detect':
            target_column = auto_target
        else:
            target_column = target_choice
            
        st.sidebar.info(f"Target: **{target_column}**")

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", 
            "🔍 EDA", 
            "⚙️ Preprocessing", 
            "📈 Results", 
            "💾 Export"
        ])

        with tab1:
            st.header("📊 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Duplicates", df.duplicated().sum())

            st.subheader("Sample Data")
            st.dataframe(df.head(10), use_container_width=True)

            # Data types
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Unique Values': df.nunique(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)

        with tab2:
            st.header("🔍 Exploratory Data Analysis")
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            if numerical_cols:
                st.subheader("📈 Numerical Columns")
                
                # Summary statistics
                st.write("**Summary Statistics:**")
                st.dataframe(df[numerical_cols].describe(), use_container_width=True)
                
                # Distribution plots
                if len(numerical_cols) > 0:
                    selected_num_col = st.selectbox("Select column for distribution plot:", numerical_cols)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_hist = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        fig_box = px.box(df, y=selected_num_col, title=f"Boxplot of {selected_num_col}")
                        st.plotly_chart(fig_box, use_container_width=True)

            if categorical_cols:
                st.subheader("📊 Categorical Columns")
                
                for col in categorical_cols:
                    st.write(f"**{col}** - {df[col].nunique()} unique values")
                    if df[col].nunique() <= 20:
                        fig = px.bar(
                            x=df[col].value_counts().index,
                            y=df[col].value_counts().values,
                            title=f"Value Counts for {col}",
                            labels={'x': col, 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Too many categories to display (>20)")
                        st.write(df[col].value_counts().head(10))

        with tab3:
            st.header("⚙️ Data Preprocessing")
            
            if st.button("🚀 Start Preprocessing", type="primary"):
                with st.spinner("Processing your data..."):
                    
                    # Step 1: Handle missing values
                    st.write("**Step 1: Handling Missing Values**")
                    if df.isnull().sum().sum() > 0:
                        df_imputed = st.session_state.preprocessor.handle_missing_values(df)
                        st.success("✅ Missing values handled using median (numerical) and mode (categorical)")
                    else:
                        df_imputed = df.copy()
                        st.info("ℹ️ No missing values found")
                    
                    # Step 2: Outlier detection
                    st.write("**Step 2: Outlier Detection**")
                    outliers = st.session_state.preprocessor.detect_outliers(df_imputed)
                    if outliers:
                        outlier_df = pd.DataFrame(outliers)
                        st.dataframe(outlier_df, use_container_width=True)
                    else:
                        st.info("ℹ️ No significant outliers detected")
                    
                    # Step 3: Scaling
                    st.write("**Step 3: Feature Scaling**")
                    scaler, scaler_reason = st.session_state.preprocessor.choose_scaler(df_imputed)
                    if scaler:
                        df_scaled = st.session_state.preprocessor.apply_scaling(df_imputed, scaler)
                        st.success(f"✅ Applied {scaler_reason}")
                    else:
                        df_scaled = df_imputed.copy()
                        st.info("ℹ️ No scaling needed")
                    
                    # Step 4: Encoding
                    st.write("**Step 4: Categorical Encoding**")
                    df_encoded, encoding_info = st.session_state.preprocessor.encode_categorical(df_scaled)
                    if encoding_info:
                        for info in encoding_info:
                            st.success(f"✅ {info}")
                    else:
                        st.info("ℹ️ No categorical columns to encode")
                    
                    # Store processed data
                    st.session_state.processed_data = df_encoded
                    
                    st.success("🎉 Preprocessing completed successfully!")

        with tab4:
            st.header("📈 Results & Recommendations")
            
            if st.session_state.processed_data is not None:
                processed_df = st.session_state.processed_data
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Shape", f"{df.shape[0]} × {df.shape[1]}")
                with col2:
                    st.metric("Processed Shape", f"{processed_df.shape[0]} × {processed_df.shape[1]}")
                
                st.subheader("🎯 ML Recommendations")
                
                # Get recommendations
                features = df.drop(columns=[target_column])
                has_target = target_column is not None
                recommendations = st.session_state.preprocessor.recommend_methods(
                    n_features=features.shape[1],
                    n_samples=df.shape[0],
                    has_target=has_target
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🔍 Feature Selection Methods:**")
                    for method in recommendations['feature_selection']:
                        st.write(f"• {method}")
                
                with col2:
                    st.write("**🤖 Recommended Models:**")
                    for model in recommendations['modeling']:
                        st.write(f"• {model}")
                
                st.subheader("📊 Processed Data Preview")
                st.dataframe(processed_df.head(10), use_container_width=True)
                
            else:
                st.info("👆 Please run preprocessing first!")

        with tab5:
            st.header("💾 Export Results")
            
            if st.session_state.processed_data is not None:
                processed_df = st.session_state.processed_data
                
                # Download processed data
                csv = processed_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Processed Data (CSV)",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
                
                # Generate summary report
                if st.button("📄 Generate Summary Report"):
                    features = df.drop(columns=[target_column])
                    recommendations = st.session_state.preprocessor.recommend_methods(
                        features.shape[1], df.shape[0], target_column is not None
                    )
                    
                    report = f"""
# AutoML Preprocessing Report

## Dataset Information
- **Original Shape:** {df.shape[0]} rows × {df.shape[1]} columns
- **Processed Shape:** {processed_df.shape[0]} rows × {processed_df.shape[1]} columns
- **Target Column:** {target_column}
- **Missing Values:** {df.isnull().sum().sum()}
- **Duplicate Rows:** {df.duplicated().sum()}

## Preprocessing Steps Applied
1. Missing value imputation (median for numerical, mode for categorical)
2. Outlier detection using IQR method
3. Feature scaling based on data characteristics
4. Categorical encoding (Label/One-hot encoding)

## ML Recommendations

### Feature Selection Methods:
{chr(10).join(['- ' + method for method in recommendations['feature_selection']])}

### Recommended Models:
{chr(10).join(['- ' + model for model in recommendations['modeling']])}

## Next Steps
1. Apply recommended feature selection methods
2. Split data into train/test sets
3. Train multiple models and compare performance
4. Optimize hyperparameters for best performing models
                    """
                    
                    st.download_button(
                        label="📄 Download Report (MD)",
                        data=report,
                        file_name="preprocessing_report.md",
                        mime="text/markdown"
                    )
                    
            else:
                st.info("👆 Please run preprocessing first!")

    else:
        st.info("👆 Please upload a CSV file to get started!")
        
        # Demo section
        st.subheader("🎯 Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🔍 Smart Analysis**")
            st.write("• Automatic target detection")
            st.write("• Missing value analysis")  
            st.write("• Outlier detection")
            st.write("• Data type inference")
        
        with col2:
            st.write("**⚙️ Intelligent Preprocessing**")
            st.write("• Smart imputation strategies")
            st.write("• Optimal scaler selection")
            st.write("• Categorical encoding")
            st.write("• Feature engineering")
        
        with col3:
            st.write("**🤖 ML Recommendations**")
            st.write("• Feature selection methods")
            st.write("• Model recommendations")
            st.write("• Pipeline optimization")
            st.write("• Export ready data")

if __name__ == "__main__":
    main()
