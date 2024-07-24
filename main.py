# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # from about_data import show_about_data
# # from data_profiling import show_data_profiling
# # from missing_value import handle_missing_values
# # from model_training import train_model
# # from outlier_detection import detect_outliers
# # from data_validation import validate_data
# # from data_imputation import impute_data

# # def main():
    # # st.set_option('deprecation.showPyplotGlobalUse', False)  # Optional configuration setting

    # # st.title("Data Dashboard")

    # # uploaded_file = st.file_uploader("Choose a file")
    # # if uploaded_file is not None:
        # # df = pd.read_csv(uploaded_file)
        
        # # option = st.sidebar.selectbox("Choose an option", ("About Data", "Data Profiling", "Missing Value Identification", "Model Training", "Outlier Detection", "Data Validation", "Data Imputation"))

        # # if option == "About Data":
            # # show_about_data(df)
        # # elif option == "Data Profiling":
            # # show_data_profiling(df)
        # # elif option == "Missing Value Identification":
            # # df = handle_missing_values(df)
        # # elif option == "Model Training":
            # # train_model(df)
        # # elif option == "Outlier Detection":
            # # detect_outliers(df)
        # # elif option == "Data Validation":
            # # validate_data(df)
        # # elif option == "Data Imputation":
            # # df = impute_data(df)

# # if __name__ == "__main__":
    # # main()
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from about_data import show_about_data
from data_profiling import show_data_profiling
from missing_value import handle_missing_values
from model_training import train_model
from outlier_detection import detect_outliers
from data_validation import validate_data
from data_imputation import impute_data

def main():
    # st.set_page_config(layout="wide")  # Set wide layout

    st.title("Data Dashboard")

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        option = st.sidebar.selectbox(
            "Choose an option", 
            ("About Data", "Data Profiling",  "Data Validation", "Data Imputation","Model Training", "Outlier Detection")###"Missing Value Identification",
        )

        if option == "About Data":
            st.subheader("About Data")
            st.write("This section provides an overview of the dataset including its head, tail, data information, and summary statistics.")
            show_about_data(df)
        elif option == "Data Profiling":
            st.subheader("Data Profiling")
            st.write("This section includes univariate and bivariate analysis of the data, along with correlation matrix and multivariate analysis.")
            st.write("**Univariate Analysis:**")
            st.write("Univariate analysis examines the distribution and summary statistics of a single variable. It helps understand the central tendency, dispersion, and shape of the variable's distribution.")
            st.write("**Bivariate Analysis:**")
            st.write("Bivariate analysis explores the relationship between two variables. It helps identify patterns, correlations, and dependencies between pairs of variables.")
            st.write("**Correlation Matrix:**")
            st.write("A correlation matrix displays the correlation coefficients between multiple variables. It helps identify the strength and direction of the relationships between variables.")
            st.write("**Multivariate Analysis:**")
            st.write("Multivariate analysis considers the interactions between multiple variables simultaneously. It helps explore complex relationships and patterns involving three or more variables.")
            show_data_profiling(df)
        # elif option == "Missing Value Identification":
            # st.subheader("Missing Value Identification")
            # st.write("This section identifies missing values in the dataset and provides options to handle them.")
            # st.write("**Missing Values:**")
            # st.write("Missing values are data points that are not recorded or available in the dataset. Identifying and handling missing values is crucial to ensure accurate and reliable analysis.")
            # df = handle_missing_values(df)
        
            
            
        elif option == "Data Validation":
            st.subheader("Data Validation")
            st.write("This section validates the data by checking for null values, duplicate rows, and additional validation tasks.")
            st.write("**Data Validation:**")
            st.write("Data validation ensures the integrity, accuracy, and consistency of the dataset. It involves checking for missing values, duplicate rows, and performing other validation tasks.")
            validate_data(df)
            
        elif option == "Model Training":
            st.subheader("Model Training")
            st.write("This section trains classification and regression models using Random Forest, Logistic Regression, and XGBoost algorithms.")
            train_model(df)
            
            
        elif option == "Outlier Detection":
            st.subheader("Outlier Detection")
            st.write("This section detects outliers in the dataset using Isolation Forest and generates an anomaly graph.")
            st.write("**Outliers:**")
            st.write("Outliers are data points that significantly deviate from the rest of the data. Detecting outliers helps identify data errors, anomalies, or interesting observations.")
            detect_outliers(df)
        
        elif option == "Data Imputation":
            st.subheader("Data Imputation")
            st.write("This section fills missing values in the dataset using mean or mode imputation.")
            st.write("**Data Imputation:**")
            st.write("Data imputation replaces missing values with estimated or calculated values to maintain the completeness of the dataset.")
            df = impute_data(df)

if __name__ == "__main__":
    main()


