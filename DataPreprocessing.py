import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

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

    def get_dataset(self):
        try:
            dataset = input("Enter the name of the dataset file (CSV): ")
            df = pd.read_csv(dataset)
            print(f"Successfully loaded dataset: {dataset}")
            return df
        except FileNotFoundError:
            print("File not found. Please check the filename.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return None

    def detect_target_column(self, df):
        """Auto-detect target column based on heuristics."""
        common_targets = ['target', 'label', 'class', 'species', 'outcome', 'y']
        for col in common_targets:
            if col in df.columns:
                print(f"Auto-detected target column: '{col}'")
                return col
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_cols) == 1:
            print(f"Auto-detected target column: '{categorical_cols[0]}' (only categorical column)")
            return categorical_cols[0]
        
        if len(categorical_cols) > 1:
            unique_counts = df[categorical_cols].nunique()
            target_candidate = unique_counts.idxmin()
            print(f"Auto-detected target column: '{target_candidate}' (categorical with least unique values)")
            return target_candidate
        
        print(f"Defaulting to last column as target: '{df.columns[-1]}'")
        return df.columns[-1]

    def sanity_check(self, df):
        if df is None:
            print("Data doesn't exist.")
            return False
            
        try:
            print("\nDATA SANITY CHECK REPORT")
            print(f"Shape of the dataset: {df.shape}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print("\nDataset Info:")
            df.info()
            print("\nMissing Values Analysis:")
            missing_percent = (df.isnull().sum() / df.shape[0] * 100).round(2)
            missing_data = pd.DataFrame({
                'Column': missing_percent.index,
                'Missing Count': df.isnull().sum(),
                'Missing %': missing_percent
            })
            missing_data = missing_data[missing_data['Missing Count'] > 0]
            if len(missing_data) > 0:
                print(missing_data.to_string(index=False))
            else:
                print("No missing values found.")
            duplicate_count = df.duplicated().sum()
            print(f"\nDuplicate rows: {duplicate_count}")
            if duplicate_count > 0:
                print(f"({duplicate_count/len(df)*100:.2f}% of total data)")
            return True
        except Exception as e:
            print(f"Error during sanity check: {e}")
            return False

    def check_categorical_distributions(self, df):
        if df is None:
            return
        obj_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(obj_cols) == 0:
            print("No categorical columns found.")
            return
        print("\nCATEGORICAL DISTRIBUTIONS")
        for col in obj_cols:
            unique_vals = df[col].nunique()
            print(f"\n'{col}' — {unique_vals} unique value(s)")
            if unique_vals <= 15:
                print(df[col].value_counts(dropna=False).head(10))
            else:
                print("High cardinality - showing top 5 values:")
                print(df[col].value_counts().head(5))

    def eda_summary(self, df):
        if df is None:
            print("Data doesn't exist.")
            return
        try:
            print("\nEXPLORATORY DATA ANALYSIS")
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                print("\nNumerical Columns Summary:")
                print(df[numerical_cols].describe().round(2))
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns
            if len(categorical_cols) > 0:
                print("\nCategorical Columns Summary:")
                print(df[categorical_cols].describe())
        except Exception as e:
            print(f"Exception occurred: {e}")

    def handle_missing_values(self, df):
        if df.isnull().sum().sum() == 0:
            print("No missing values to handle.")
            return df
        print("\nHandling Missing Values...")
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        df_processed = df.copy()
        if self.numerical_cols:
            num_imputer = SimpleImputer(strategy='median')
            df_processed[self.numerical_cols] = num_imputer.fit_transform(df[self.numerical_cols])
            print(f"Imputed {len(self.numerical_cols)} numerical columns with median")
        if self.categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_processed[self.categorical_cols] = cat_imputer.fit_transform(df[self.categorical_cols])
            print(f"Imputed {len(self.categorical_cols)} categorical columns with mode")
        return df_processed

    def detect_and_report_outliers(self, df):
        outlier_summary = []
        for col in df.select_dtypes(include=[np.number]).columns:
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
        if outlier_summary:
            print("\nOutlier Detection Results:")
            outlier_df = pd.DataFrame(outlier_summary)
            print(outlier_df.to_string(index=False))
            return [item['Column'] for item in outlier_summary]
        else:
            print("No significant outliers found.")
            return []

    def choose_and_apply_scaler(self, df):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            print("No numerical columns to scale.")
            return df
        outlier_cols = []
        sparse_cols = []
        skewed_cols = []
        for col in numerical_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(df) * 0.01:
                outlier_cols.append(col)
            zero_ratio = (df[col] == 0).sum() / len(df)
            if zero_ratio > 0.5:
                sparse_cols.append(col)
            if abs(df[col].skew()) > 1:
                skewed_cols.append(col)
        scaler_choice = None
        if len(outlier_cols) > 0:
            scaler_choice = "RobustScaler"
            self.scaler = RobustScaler()
        elif len(sparse_cols) > 0:
            scaler_choice = "MaxAbsScaler"
            self.scaler = MaxAbsScaler()
        else:
            scaler_choice = "StandardScaler"
            self.scaler = StandardScaler()
        print(f"\nScaling data using {scaler_choice}")
        df_scaled = df.copy()
        df_scaled[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        return df_scaled

    def encode_categorical(self, df):
        df_encoded = df.copy()
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(self.categorical_cols) == 0:
            print("No categorical columns to encode.")
            return df_encoded
        print("\nEncoding Categorical Columns...")
        for col in self.categorical_cols:
            unique_vals = df[col].nunique()
            if unique_vals <= 2:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col])
                self.encoders[col] = le
                print(f"Label encoded '{col}' (binary)")
            else:
                ohe = OneHotEncoder(sparse_output=False, drop='first')
                transformed = ohe.fit_transform(df[[col]])
                new_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                df_ohe = pd.DataFrame(transformed, columns=new_cols, index=df.index)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), df_ohe], axis=1)
                self.encoders[col] = ohe
                print(f"One-hot encoded '{col}' with {unique_vals} unique values")
        return df_encoded

    def preprocess(self, df):
        if df is None:
            return None
        target_col = self.detect_target_column(df)
        features = df.drop(columns=[target_col])
        target = df[target_col]

        print(f"\nTarget column detected: '{target_col}'")
        print(f"Number of features: {features.shape[1]}, Number of samples: {df.shape[0]}")

        if not self.sanity_check(df):
            return None
        self.check_categorical_distributions(df)
        self.eda_summary(df)

        df_imputed = self.handle_missing_values(df)
        outlier_cols = self.detect_and_report_outliers(df_imputed)
        df_scaled = self.choose_and_apply_scaler(df_imputed)
        df_encoded = self.encode_categorical(df_scaled)

        has_target = target_col is not None and target_col in df.columns
        recs = self.recommend_methods(
            n_features=features.shape[1],
            n_samples=df.shape[0],
            has_target=has_target
        )
        print("\nRECOMMENDATIONS")
        print(f"Feature Selection Methods: {recs['feature_selection']}")
        print(f"Modeling Techniques: {recs['modeling']}\n")

        return df_encoded, target_col

# Usage example:
if __name__ == "__main__":
    dp = DataPreprocessor()
    df = dp.get_dataset()
    if df is not None:
        processed_df, target_col = dp.preprocess(df)
