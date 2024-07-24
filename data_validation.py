import streamlit as st
import pandas as pd
import numpy as np
# def validate_data(df):
    # st.header("Data Validation")

    # st.subheader("Checking for null values")
    # if df.isnull().sum().sum() == 0:
        # st.write("No null values detected.")
    # else:
        # st.write("Null values detected.")
        # st.write(df.isnull().sum())

    # st.subheader("Checking for duplicate rows")
    # duplicate_rows = df[df.duplicated(keep=False)]
    # if len(duplicate_rows) == 0:
        # st.write("No duplicate rows detected.")
    # else:
        # st.write("Duplicate rows detected. Count:", len(duplicate_rows))
        # st.write("Top 5 duplicate rows:")
        # st.dataframe(duplicate_rows.head())

    # # Additional functionality
    # st.subheader("Additional Functionality")
    # option = st.selectbox("Choose an additional validation task", ("Check for Constant Columns", "Check for Unique Values in Columns"))
    # if option == "Check for Constant Columns":
        # constant_columns = df.columns[df.nunique() == 1]
        # if len(constant_columns) == 0:
            # st.write("No constant columns detected.")
        # else:
            # st.write("Constant columns detected:")
            # st.write(constant_columns)
    # elif option == "Check for Unique Values in Columns":
        # selected_column = st.selectbox("Select column to check for unique values", df.columns)
        # unique_values = df[selected_column].unique()
        # st.write("Unique values in selected column:")
        # st.write(unique_values)


# def validate_data(df):
    # st.header("Data Validation")

    # # Dropdown menu for validation options
    # validation_option = st.selectbox("Select Validation Option", 
                                     # ["Data Type Checks", "Range Checks for Numeric Columns", 
                                      # "Unique Value Checks for Categorical Columns", 
                                      # "Unique Value Checks for Numeric Columns", 
                                      # "Missing Values Check", "Duplicate Rows Check",
                                      # "Outlier Detection for Numeric Columns", "Check for Constant Columns", 
                                      # "Check for Column Data Type Mismatches", "Check for Specific Value Constraints"])

    # if validation_option == "Data Type Checks":
        # st.subheader("Data Type Checks")
        # data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
        # st.dataframe(data_types)

    # elif validation_option == "Range Checks for Numeric Columns":
        # st.subheader("Range Checks for Numeric Columns")
        # numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # if numeric_columns:
            # range_data = pd.DataFrame({
                # "Column": numeric_columns,
                # "Min": [df[col].min() for col in numeric_columns],
                # "Max": [df[col].max() for col in numeric_columns]
            # })
            # st.dataframe(range_data)
        # else:
            # st.write("No numeric columns found.")

    # elif validation_option == "Unique Value Checks for Categorical Columns":
        # st.subheader("Unique Value Checks for Categorical Columns")
        # categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        # if categorical_columns:
            # unique_data = pd.DataFrame({
                # "Column": categorical_columns,
                # "Unique Values": [df[col].nunique() for col in categorical_columns]
            # })
            # st.dataframe(unique_data)
        # else:
            # st.write("No categorical columns found.")

    # elif validation_option == "Unique Value Checks for Numeric Columns":
        # st.subheader("Unique Value Checks for Numeric Columns")
        # numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # if numeric_columns:
            # unique_data = pd.DataFrame({
                # "Column": numeric_columns,
                # "Unique Values": [df[col].nunique() for col in numeric_columns]
            # })
            # st.dataframe(unique_data)
        # else:
            # st.write("No numeric columns found.")

    # elif validation_option == "Missing Values Check":
        # st.subheader("Missing Values Check")
        # missing_values = df.isnull().sum()
        # missing_data = pd.DataFrame(missing_values, columns=['Missing Values']).reset_index()
        # missing_data.columns = ['Column', 'Missing Values']
        # st.dataframe(missing_data)

    # elif validation_option == "Duplicate Rows Check":
        # st.subheader("Duplicate Rows Check")
        # duplicate_rows = df[df.duplicated(keep=False)]
        # if len(duplicate_rows) == 0:
            # st.write("No duplicate rows detected.")
        # else:
            # st.write(f"Duplicate rows detected: {len(duplicate_rows)}")
            # st.write("Top 5 duplicate rows:")
            # st.dataframe(duplicate_rows.head())

    # elif validation_option == "Outlier Detection for Numeric Columns":
        # st.subheader("Outlier Detection for Numeric Columns")
        # numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # selected_column = st.selectbox("Select column for outlier detection", numeric_columns)
        # if selected_column:
            # Q1 = df[selected_column].quantile(0.25)
            # Q3 = df[selected_column].quantile(0.75)
            # IQR = Q3 - Q1
            # outliers = df[(df[selected_column] < (Q1 - 1.5 * IQR)) | (df[selected_column] > (Q3 + 1.5 * IQR))]
            # st.write(f"Outliers detected: {len(outliers)}")
            # st.write("Top 5 outliers:")
            # st.dataframe(outliers.head())

    # elif validation_option == "Check for Constant Columns":
        # st.subheader("Check for Constant Columns")
        # constant_columns = [col for col in df.columns if df[col].nunique() == 1]
        # if constant_columns:
            # st.write("Constant columns detected:")
            # st.write(constant_columns)
        # else:
            # st.write("No constant columns detected.")

    # elif validation_option == "Check for Column Data Type Mismatches":
        # st.subheader("Check for Column Data Type Mismatches")
        # mismatches = {}
        # for column in df.columns:
            # if pd.api.types.is_numeric_dtype(df[column]):
                # mismatches[column] = df[column].apply(lambda x: isinstance(x, (int, float))).sum() != len(df[column])
            # elif pd.api.types.is_string_dtype(df[column]):
                # mismatches[column] = df[column].apply(lambda x: isinstance(x, str)).sum() != len(df[column])
        # mismatches = {k: v for k, v in mismatches.items() if v}
        # if mismatches:
            # st.write("Data type mismatches detected:")
            # st.write(mismatches)
        # else:
            # st.write("No data type mismatches detected.")

    # elif validation_option == "Check for Specific Value Constraints":
        # st.subheader("Check for Specific Value Constraints")
        # selected_column = st.selectbox("Select column to check for specific values", df.columns)
        # if selected_column:
            # unique_values = df[selected_column].unique()
            # st.write(f"Unique values in column '{selected_column}':")
            # st.write(unique_values)



# import streamlit as st
# import pandas as pd
# import base64  # Needed for encoding the CSV file for download

# def validate_data(df):
    # st.header("Data Validation")

    # # Dropdown menu for validation options
    # validation_option = st.selectbox("Select Validation Option", 
                                     # ["Data Type Checks", "Range Checks for Numeric Columns", 
                                      # "Unique Value Checks for Categorical Columns", 
                                      # "Unique Value Checks for Numeric Columns", 
                                      # "Missing Values Check", "Duplicate Rows Check",
                                      # "Outlier Detection for Numeric Columns", "Check for Constant Columns", 
                                      # "Check for Column Data Type Mismatches", "Check for Specific Value Constraints"])

    # if validation_option == "Data Type Checks":
        # st.subheader("Data Type Checks")
        # data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
        # st.dataframe(data_types)

    # elif validation_option == "Range Checks for Numeric Columns":
        # st.subheader("Range Checks for Numeric Columns")
        # numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # if numeric_columns:
            # range_data = pd.DataFrame({
                # "Column": numeric_columns,
                # "Min": [df[col].min() for col in numeric_columns],
                # "Max": [df[col].max() for col in numeric_columns]
            # })
            # st.dataframe(range_data)
        # else:
            # st.write("No numeric columns found.")

    # elif validation_option == "Unique Value Checks for Categorical Columns":
        # st.subheader("Unique Value Checks for Categorical Columns")
        # categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        # if categorical_columns:
            # unique_data = pd.DataFrame({
                # "Column": categorical_columns,
                # "Unique Values": [df[col].nunique() for col in categorical_columns]
            # })
            # st.dataframe(unique_data)
        # else:
            # st.write("No categorical columns found.")

    # elif validation_option == "Unique Value Checks for Numeric Columns":
        # st.subheader("Unique Value Checks for Numeric Columns")
        # numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # if numeric_columns:
            # unique_data = pd.DataFrame({
                # "Column": numeric_columns,
                # "Unique Values": [df[col].nunique() for col in numeric_columns]
            # })
            # st.dataframe(unique_data)
        # else:
            # st.write("No numeric columns found.")

    # elif validation_option == "Missing Values Check":
        # st.subheader("Missing Values Check")
        # missing_values = df.isnull().sum()
        # missing_data = pd.DataFrame(missing_values, columns=['Missing Values']).reset_index()
        # missing_data.columns = ['Column', 'Missing Values']
        # st.dataframe(missing_data)

    # elif validation_option == "Duplicate Rows Check":
        # st.subheader("Duplicate Rows Check")
        # duplicate_rows = df[df.duplicated(keep=False)]
        # if len(duplicate_rows) == 0:
            # st.write("No duplicate rows detected.")
        # else:
            # st.write(f"Duplicate rows detected: {len(duplicate_rows)}")
            # st.write("Top 5 duplicate rows:")
            # st.dataframe(duplicate_rows.head())

            # # Data download option for duplicate rows
            # if st.button("Download All Duplicate Rows as CSV"):
                # csv = duplicate_rows.to_csv(index=False)
                # b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding for download
                # href = f'<a href="data:file/csv;base64,{b64}" download="duplicate_rows.csv">Download CSV File</a>'
                # st.markdown(href, unsafe_allow_html=True)

    # elif validation_option == "Outlier Detection for Numeric Columns":
        # st.subheader("Outlier Detection for Numeric Columns")
        # numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # selected_column = st.selectbox("Select column for outlier detection", numeric_columns)
        # if selected_column:
            # Q1 = df[selected_column].quantile(0.25)
            # Q3 = df[selected_column].quantile(0.75)
            # IQR = Q3 - Q1
            # outliers = df[(df[selected_column] < (Q1 - 1.5 * IQR)) | (df[selected_column] > (Q3 + 1.5 * IQR))]
            # st.write(f"Outliers detected: {len(outliers)}")
            # st.write("Top 5 outliers:")
            # st.dataframe(outliers.head())

    # elif validation_option == "Check for Constant Columns":
        # st.subheader("Check for Constant Columns")
        # constant_columns = [col for col in df.columns if df[col].nunique() == 1]
        # if constant_columns:
            # st.write("Constant columns detected:")
            # st.write(constant_columns)
        # else:
            # st.write("No constant columns detected.")

    # elif validation_option == "Check for Column Data Type Mismatches":
        # st.subheader("Check for Column Data Type Mismatches")
        # mismatches = {}
        # for column in df.columns:
            # if pd.api.types.is_numeric_dtype(df[column]):
                # mismatches[column] = df[column].apply(lambda x: isinstance(x, (int, float))).sum() != len(df[column])
            # elif pd.api.types.is_string_dtype(df[column]):
                # mismatches[column] = df[column].apply(lambda x: isinstance(x, str)).sum() != len(df[column])
        # mismatches = {k: v for k, v in mismatches.items() if v}
        # if mismatches:
            # st.write("Data type mismatches detected:")
            # st.write(mismatches)
        # else:
            # st.write("No data type mismatches detected.")

    # elif validation_option == "Check for Specific Value Constraints":
        # st.subheader("Check for Specific Value Constraints")
        # selected_column = st.selectbox("Select column to check for specific values", df.columns)
        # if selected_column:
            # unique_values = df[selected_column].unique()
            # st.write(f"Unique values in column '{selected_column}':")
            # st.write(unique_values)






import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64  # Needed for encoding the CSV file for download

def validate_data(df):
    st.header("Data Validation")

    # Dropdown menu for validation options
    validation_option = st.selectbox("Select Validation Option", 
                                     ["Data Type Checks", "Range Checks for Numeric Columns", 
                                      "Unique Value Checks for Categorical Columns", 
                                      "Unique Value Checks for Numeric Columns", 
                                      "Missing Values Check", "Duplicate Rows Check",
                                      "Outlier Detection for Numeric Columns", "Check for Constant Columns", 
                                      "Check for Column Data Type Mismatches", "Check for Specific Value Constraints"])

    if validation_option == "Data Type Checks":
        st.subheader("Data Type Checks")
        data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
        st.dataframe(data_types)

    elif validation_option == "Range Checks for Numeric Columns":
        st.subheader("Range Checks for Numeric Columns")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_columns:
            range_data = pd.DataFrame({
                "Column": numeric_columns,
                "Min": [df[col].min() for col in numeric_columns],
                "Max": [df[col].max() for col in numeric_columns]
            })
            st.dataframe(range_data)
            
            # Generate plots
            fig, ax = plt.subplots()
            sns.boxplot(data=df[numeric_columns], orient='h', ax=ax)
            ax.set_title('Range Check Boxplot for Numeric Columns')
            st.pyplot(fig)
        else:
            st.write("No numeric columns found.")

    elif validation_option == "Unique Value Checks for Categorical Columns":
        st.subheader("Unique Value Checks for Categorical Columns")
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            unique_data = pd.DataFrame({
                "Column": categorical_columns,
                "Unique Values": [df[col].nunique() for col in categorical_columns]
            })
            st.dataframe(unique_data)
            
            # Generate plots
            for col in categorical_columns:
                fig, ax = plt.subplots()
                sns.countplot(y=col, data=df, ax=ax)
                ax.set_title(f'Unique Value Count for {col}')
                st.pyplot(fig)
        else:
            st.write("No categorical columns found.")

    elif validation_option == "Unique Value Checks for Numeric Columns":
        st.subheader("Unique Value Checks for Numeric Columns")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_columns:
            unique_data = pd.DataFrame({
                "Column": numeric_columns,
                "Unique Values": [df[col].nunique() for col in numeric_columns]
            })
            st.dataframe(unique_data)
            
            # Generate plots
            for col in numeric_columns:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f'Unique Value Distribution for {col}')
                st.pyplot(fig)
        else:
            st.write("No numeric columns found.")

    elif validation_option == "Missing Values Check":
        st.subheader("Missing Values Check")
        missing_values = df.isnull().sum()
        missing_data = pd.DataFrame(missing_values, columns=['Missing Values']).reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        st.dataframe(missing_data)
        
        # Generate plots
        fig, ax = plt.subplots()
        sns.barplot(x='Missing Values', y='Column', data=missing_data, ax=ax)
        ax.set_title('Missing Values Count by Column')
        st.pyplot(fig)

    elif validation_option == "Duplicate Rows Check":
        st.subheader("Duplicate Rows Check")
        duplicate_rows = df[df.duplicated(keep=False)]
        if len(duplicate_rows) == 0:
            st.write("No duplicate rows detected.")
        else:
            st.write(f"Duplicate rows detected: {len(duplicate_rows)}")
            st.write("Top 5 duplicate rows:")
            st.dataframe(duplicate_rows.head())

            # Data download option for duplicate rows
            if st.button("Download All Duplicate Rows as CSV"):
                csv = duplicate_rows.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding for download
                href = f'<a href="data:file/csv;base64,{b64}" download="duplicate_rows.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

    elif validation_option == "Outlier Detection for Numeric Columns":
        st.subheader("Outlier Detection for Numeric Columns")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_column = st.selectbox("Select column for outlier detection", numeric_columns)
        if selected_column:
            Q1 = df[selected_column].quantile(0.25)
            Q3 = df[selected_column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[selected_column] < (Q1 - 1.5 * IQR)) | (df[selected_column] > (Q3 + 1.5 * IQR))]
            st.write(f"Outliers detected: {len(outliers)}")
            st.write("Top 5 outliers:")
            st.dataframe(outliers.head())
            
            # Generate plots
            fig, ax = plt.subplots()
            sns.boxplot(x=selected_column, data=df, ax=ax)
            ax.set_title(f'Boxplot for {selected_column}')
            st.pyplot(fig)

    elif validation_option == "Check for Constant Columns":
        st.subheader("Check for Constant Columns")
        constant_columns = [col for col in df.columns if df[col].nunique() == 1]
        if constant_columns:
            st.write("Constant columns detected:")
            st.write(constant_columns)
        else:
            st.write("No constant columns detected.")

    elif validation_option == "Check for Column Data Type Mismatches":
        st.subheader("Check for Column Data Type Mismatches")
        mismatches = {}
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                mismatches[column] = df[column].apply(lambda x: isinstance(x, (int, float))).sum() != len(df[column])
            elif pd.api.types.is_string_dtype(df[column]):
                mismatches[column] = df[column].apply(lambda x: isinstance(x, str)).sum() != len(df[column])
        mismatches = {k: v for k, v in mismatches.items() if v}
        if mismatches:
            st.write("Data type mismatches detected:")
            st.write(mismatches)
        else:
            st.write("No data type mismatches detected.")

    elif validation_option == "Check for Specific Value Constraints":
        st.subheader("Check for Specific Value Constraints")
        selected_column = st.selectbox("Select column to check for specific values", df.columns)
        if selected_column:
            unique_values = df[selected_column].unique()
            st.write(f"Unique values in column '{selected_column}':")
            st.write(unique_values)
            
            # Generate plots
            fig, ax = plt.subplots()
            sns.countplot(y=selected_column, data=df, ax=ax)
            ax.set_title(f'Unique Values Count for {selected_column}')
            st.pyplot(fig)

