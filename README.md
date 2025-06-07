# 🤖 AutoML Data Preprocessor

A comprehensive web-based application for intelligent data preprocessing and machine learning pipeline recommendations. This tool automates the entire data preprocessing workflow, from data analysis to ML model recommendations, making machine learning accessible to users of all skill levels.

## ✨ Features

### 🔍 Smart Data Analysis
- **Automatic target detection** using intelligent heuristics
- **Comprehensive data profiling** with missing values, duplicates, and data type analysis
- **Advanced outlier detection** using IQR-based statistical methods
- **Interactive exploratory data analysis** with dynamic visualizations

### ⚙️ Intelligent Preprocessing
- **Smart missing value imputation** with strategy selection based on data characteristics
- **Optimal scaler selection** (StandardScaler, RobustScaler, MaxAbsScaler) based on data distribution
- **Intelligent categorical encoding** (Label encoding for binary, One-hot encoding for multi-class)
- **High cardinality handling** with warnings and appropriate encoding strategies

### 🤖 ML Recommendations
- **Automated feature selection method recommendations** based on dataset characteristics
- **Intelligent model suggestions** tailored to data size and complexity
- **Pipeline optimization suggestions** for different scenarios (supervised/unsupervised learning)

### 📊 Interactive Visualizations
- **Distribution plots** and box plots for numerical features
- **Value count visualizations** for categorical features
- **Real-time data profiling** with comprehensive statistics
- **Before/after preprocessing comparisons**

### 💾 Export & Reporting
- **Download processed datasets** in CSV format
- **Generate comprehensive preprocessing reports** in Markdown
- **Summary statistics** and transformation logs
- **Ready-to-use data** for ML pipelines

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/automl-data-preprocessor.git
cd automl-data-preprocessor
```

2. **Install required packages**
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly
```

### Usage

1. **Launch the application**
```bash
streamlit run app.py
```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload your CSV file** using the sidebar file uploader

4. **Explore your data** through the interactive tabs:
   - **Data Overview**: Basic statistics and data profiling
   - **EDA**: Interactive exploratory data analysis
   - **Preprocessing**: Automated data preprocessing pipeline
   - **Results**: ML recommendations and processed data preview
   - **Export**: Download processed data and reports

## 📋 How It Works

### 1. Data Analysis Phase
- Automatically detects target columns using common naming patterns
- Analyzes data types, missing values, and statistical distributions
- Identifies outliers using the Interquartile Range (IQR) method
- Provides comprehensive data profiling and quality assessment

### 2. Preprocessing Pipeline
```python
# The preprocessing pipeline follows these steps:
1. Missing Value Imputation → Median for numerical, Mode for categorical
2. Outlier Detection → IQR-based statistical analysis
3. Feature Scaling → Intelligent scaler selection based on data characteristics
4. Categorical Encoding → Binary: Label encoding, Multi-class: One-hot encoding
```

### 3. ML Recommendations Engine
The system provides intelligent recommendations based on:
- **Dataset size** (samples and features)
- **Data complexity** and characteristics
- **Target variable presence** (supervised vs unsupervised)
- **Feature dimensionality** considerations

## 🎯 Supported Data Types

- **Numerical features**: Integer, Float, Continuous variables
- **Categorical features**: String, Object, Categorical variables
- **Mixed datasets**: Combination of numerical and categorical features
- **Various scales**: Different ranges and distributions

## 📊 Preprocessing Capabilities

### Missing Value Handling
- **Numerical**: Median imputation (robust to outliers)
- **Categorical**: Most frequent value imputation
- **Automatic detection** and handling of missing patterns

### Feature Scaling
- **StandardScaler**: For normally distributed data
- **RobustScaler**: For data with outliers
- **MaxAbsScaler**: For sparse data with many zeros
- **Intelligent selection** based on data characteristics

### Encoding Strategies
- **Binary categorical**: Label encoding (0/1)
- **Multi-class categorical**: One-hot encoding with drop='first'
- **High cardinality warning**: For features with >20 categories
- **Memory-efficient** sparse matrix handling

## 🤖 ML Recommendations

### Feature Selection Methods
- **Variance Threshold**: Remove low-variance features
- **Univariate Selection**: Statistical tests (chi2, ANOVA)
- **Mutual Information**: Capture non-linear relationships
- **Recursive Feature Elimination**: Model-based selection
- **Sequential Feature Selection**: Forward/backward selection

### Model Recommendations
Based on dataset characteristics:
- **Small datasets**: SVM, KNN, Linear models
- **Large datasets**: Random Forest, XGBoost, LightGBM
- **High dimensionality**: Regularized models (Ridge, Lasso)
- **Unsupervised**: KMeans, PCA, Autoencoders

## 📈 Example Workflow

1. **Upload** your CSV dataset
2. **Review** automatic target detection or select manually
3. **Explore** your data through interactive EDA
4. **Run** the automated preprocessing pipeline
5. **Review** ML recommendations and processed data
6. **Export** cleaned data and preprocessing report
7. **Implement** recommended ML models and feature selection methods

## 🛠️ Technical Architecture

### Core Components
```
DataPreprocessor Class
├── Missing Value Handler
├── Outlier Detector  
├── Scaler Selector
├── Categorical Encoder
└── ML Recommender

Streamlit Interface
├── Data Upload & Validation
├── Interactive EDA
├── Preprocessing Pipeline
├── Results Visualization
└── Export Functionality
```

### Key Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Preprocessing and ML utilities
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical computing

## 🎨 User Interface

The application features a clean, intuitive interface with:
- **Responsive design** that works on desktop and mobile
- **Interactive tabs** for organized workflow
- **Real-time feedback** during preprocessing
- **Professional visualizations** with Plotly
- **Download capabilities** for all outputs

## 🔄 Future Enhancements

- **Advanced feature engineering** (polynomial features, interactions)
- **Time series preprocessing** capabilities
- **Text data preprocessing** (NLP pipeline)
- **Image data preprocessing** support
- **Integration with popular ML frameworks** (XGBoost, LightGBM)
- **Automated hyperparameter tuning** recommendations
- **Model performance comparison** tools

## 📝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn** team for comprehensive preprocessing tools
- **Streamlit** team for the amazing web app framework
- **Plotly** team for interactive visualization capabilities
- The **open-source community** for continuous inspiration and support

## 📞 Support

If you encounter any issues or have questions:
- **Open an issue** on GitHub
- **Check the documentation** for troubleshooting tips
- **Review example datasets** and workflows

---

**Made with ❤️ for the ML community**

*Transform your raw data into ML-ready datasets with just a few clicks!*
