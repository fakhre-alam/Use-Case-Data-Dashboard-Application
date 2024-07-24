# import streamlit as st
# import pandas as pd
# import numpy as np
# def impute_data(df):
    # st.header("Data Imputation and Enrichment")
    # st.subheader("Filling missing values with mean or mode")

    # numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    # categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # if st.checkbox("Fill missing values with mean for numeric columns"):
        # df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        # st.write("Missing values filled for numeric columns.")

    # if st.checkbox("Fill missing values with mode for categorical columns"):
        # df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        # st.write("Missing values filled for categorical columns.")

    # return df


# import streamlit as st
# import pandas as pd
# import numpy as np

# def impute_data(df):
    # st.header("Data Imputation and Enrichment")
    
    # # Identify columns with missing values
    # missing_data = df[df.isnull().any(axis=1)]
    # st.subheader("Rows with Missing Values")
    # if not missing_data.empty:
        # st.write("Top 5 rows with missing values before imputation:")
        # st.dataframe(missing_data.head())
    # else:
        # st.write("No missing values found.")
    
    # # Columns classification
    # numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    # categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # # Imputation
    # st.subheader("Filling missing values with mean or mode")
    # if st.checkbox("Fill missing values with mean for numeric columns"):
        # df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        # st.write("Missing values filled for numeric columns.")

    # if st.checkbox("Fill missing values with mode for categorical columns"):
        # df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        # st.write("Missing values filled for categorical columns.")
    
    # # Display rows after imputation
    # st.subheader("Rows After Imputation")
    # if not df.empty:
        # st.write("Top 5 rows after imputation:")
        # st.dataframe(df.head())

    # return df
    
    
    
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder

# def impute_data(df):
    # st.header("Data Imputation and Enrichment")
    
    # # Identify columns with missing values
    # missing_data = df[df.isnull().any(axis=1)]
    # st.subheader("Rows with Missing Values")
    # if not missing_data.empty:
        # st.write("Top 5 rows with missing values before imputation:")
        # st.dataframe(missing_data.head())
    # else:
        # st.write("No missing values found.")
    
    # # Columns classification
    # numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    # categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # # Imputation
    # st.subheader("Filling missing values with mean or mode")
    # if st.checkbox("Fill missing values with mean for numeric columns"):
        # df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        # st.write("Missing values filled for numeric columns.")

    # if st.checkbox("Fill missing values with mode for categorical columns"):
        # df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        # st.write("Missing values filled for categorical columns.")
    
    # if st.checkbox("Fill missing values with Random Forest model"):
        # st.write("Using Random Forest model for missing value imputation.")
        
        # # Encoding categorical variables
        # le_dict = {}
        # for column in categorical_columns:
            # le = LabelEncoder()
            # df[column] = le.fit_transform(df[column].astype(str))
            # le_dict[column] = le
        
        # # Define imputer
        # imputer = IterativeImputer(estimator=RandomForestRegressor(), missing_values=np.nan, max_iter=10, random_state=0)
        # df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        # # Decoding categorical variables
        # for column in categorical_columns:
            # df_imputed[column] = df_imputed[column].round().astype(int)
            # df_imputed[column] = le_dict[column].inverse_transform(df_imputed[column])
        
        # df = df_imputed
        # st.write("Missing values imputed using Random Forest model.")

    # # Display rows after imputation
    # st.subheader("Rows After Imputation")
    # if not df.empty:
        # st.write("Top 5 rows after imputation:")
        # st.dataframe(df.head())

    # return df

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder

# def impute_data(df):
    # st.header("Data Imputation and Enrichment")
    
    # # Identify columns with missing values
    # missing_data = df[df.isnull().any(axis=1)]
    # st.subheader("Rows with Missing Values")
    # if not missing_data.empty:
        # st.write("Top 5 rows with missing values before imputation:")
        # st.dataframe(missing_data.head())
    # else:
        # st.write("No missing values found.")
    
    # # Columns classification
    # numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    # categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # # Imputation based on user selection
    # st.subheader("Filling missing values")
    
    # if st.checkbox("Fill missing values with mean for numeric columns"):
        # df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        # st.write("Missing values filled for numeric columns.")
        # st.write("Top 5 rows after imputation:")
        # st.dataframe(df.head())
    
    # if st.checkbox("Fill missing values with mode for categorical columns"):
        # for col in categorical_columns:
            # df[col] = df[col].fillna(df[col].mode()[0])
        # st.write("Missing values filled for categorical columns.")
        # st.write("Top 5 rows after imputation:")
        # st.dataframe(df.head())
    
    # if st.checkbox("Fill missing values with Random Forest model for categorical columns"):
        # if not categorical_columns:
            # st.write("No categorical columns found.")
        # else:
            # # Encoding categorical variables
            # le_dict = {}
            # for column in categorical_columns:
                # le = LabelEncoder()
                # df[column] = le.fit_transform(df[column].astype(str))
                # le_dict[column] = le
            
            # # Define imputer
            # imputer = IterativeImputer(estimator=RandomForestRegressor(), missing_values=np.nan, max_iter=10, random_state=0)
            # df_imputed = pd.DataFrame(imputer.fit_transform(df[categorical_columns]), columns=categorical_columns)
            
            # # Decoding categorical variables
            # for column in categorical_columns:
                # df_imputed[column] = df_imputed[column].round().astype(int)
                # df_imputed[column] = le_dict[column].inverse_transform(df_imputed[column])
            
            # df[categorical_columns] = df_imputed
            # st.write("Missing values imputed using Random Forest model for categorical columns.")
            # st.write("Top 5 rows after imputation:")
            # st.dataframe(df.head())
    
    # if st.checkbox("Fill missing values with Random Forest model for both numeric and categorical columns"):
        # # Encoding categorical variables
        # le_dict = {}
        # for column in categorical_columns:
            # le = LabelEncoder()
            # df[column] = le.fit_transform(df[column].astype(str))
            # le_dict[column] = le
        
        # # Define imputer for both numeric and categorical columns
        # imputer_numeric = IterativeImputer(estimator=RandomForestRegressor(), missing_values=np.nan, max_iter=10, random_state=0)
        # df_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(df[numeric_columns]), columns=numeric_columns)
        
        # imputer_categorical = IterativeImputer(estimator=RandomForestRegressor(), missing_values=np.nan, max_iter=10, random_state=0)
        # df_categorical_imputed = pd.DataFrame(imputer_categorical.fit_transform(df[categorical_columns]), columns=categorical_columns)
        
        # # Decoding categorical variables
        # for column in categorical_columns:
            # df_categorical_imputed[column] = df_categorical_imputed[column].round().astype(int)
            # df_categorical_imputed[column] = le_dict[column].inverse_transform(df_categorical_imputed[column])
        
        # df[numeric_columns] = df_numeric_imputed
        # df[categorical_columns] = df_categorical_imputed
        
        # st.write("Missing values imputed using Random Forest model for both numeric and categorical columns.")
        # st.write("Top 5 rows after imputation:")
        # st.dataframe(df.head())

    # return df




import streamlit as st
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import base64  # Needed for encoding the CSV file for download

def impute_data(df):
    st.header("Data Imputation and Enrichment")
    
    # Identify columns with missing values
    missing_columns = df.columns[df.isnull().any()].tolist()
    missing_data = df[df.isnull().any(axis=1)]
    st.subheader("Rows with Missing Values")
    if not missing_data.empty:
        st.write("Top 5 rows with missing values before imputation:")
        st.dataframe(missing_data.head())
    else:
        st.write("No missing values found.")
    # Columns classification
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Separate numeric and categorical columns with missing values
    numeric_columns_with_missing = df[numeric_columns].columns[df[numeric_columns].isnull().any()].tolist()
    categorical_columns_with_missing = df[categorical_columns].columns[df[categorical_columns].isnull().any()].tolist()

    # Track imputation counts and columns
    num_imputed_numeric = 0
    num_imputed_categorical = 0
    imputed_columns = []
    st.subheader("Missing Values Check")
    missing_values = df.isnull().sum()
    missing_data = pd.DataFrame(missing_values, columns=['Missing Values']).reset_index()
    missing_data.columns = ['Column', 'Missing Values']
    st.dataframe(missing_data)
    # Imputation based on user selection
    st.subheader("Filling missing values")
    
    if st.checkbox("Fill missing values with mean for numeric columns") and numeric_columns_with_missing:
        num_imputed_numeric += df[numeric_columns_with_missing].isnull().sum().sum()
        df[numeric_columns_with_missing] = df[numeric_columns_with_missing].fillna(df[numeric_columns_with_missing].mean())
        imputed_columns.extend(numeric_columns_with_missing)
        st.write("Missing values filled for numeric columns.")
        st.write(f"Number of rows imputed for numeric columns: {num_imputed_numeric}")
        st.write(f"Columns imputed: {', '.join(numeric_columns_with_missing)}")
        st.write("Top 5 rows after imputation:")
        st.dataframe(df.head())
        
        
        # Data download option for mean imputation
        if st.button("Download CSV after Mean Imputation"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding for download
            href = f'<a href="data:file/csv;base64,{b64}" download="mean_imputed_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    if st.checkbox("Fill missing values with mode for categorical columns") and categorical_columns_with_missing:
        num_imputed_categorical += df[categorical_columns_with_missing].isnull().sum().sum()
        for col in categorical_columns_with_missing:
            df[col] = df[col].fillna(df[col].mode()[0])
        imputed_columns.extend(categorical_columns_with_missing)
        st.write("Missing values filled for categorical columns.")
        st.write(f"Number of rows imputed for categorical columns: {num_imputed_categorical}")
        st.write(f"Columns imputed: {', '.join(categorical_columns_with_missing)}")
        st.write("Top 5 rows after imputation:")
        st.dataframe(df.head())
        
        # Data download option for mode imputation
        if st.button("Download CSV after Mode Imputation"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding for download
            href = f'<a href="data:file/csv;base64,{b64}" download="mode_imputed_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    if st.checkbox("Fill missing values with Random Forest model for categorical columns") and categorical_columns_with_missing:
        num_imputed_categorical += df[categorical_columns_with_missing].isnull().sum().sum()
        # Encoding categorical variables
        le_dict = {}
        for column in categorical_columns_with_missing:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            le_dict[column] = le
        
        # Define imputer
        imputer = IterativeImputer(estimator=RandomForestRegressor(), missing_values=np.nan, max_iter=10, random_state=0)
        df_imputed = pd.DataFrame(imputer.fit_transform(df[categorical_columns_with_missing]), columns=categorical_columns_with_missing)
        
        # Decoding categorical variables
        for column in categorical_columns_with_missing:
            df_imputed[column] = df_imputed[column].round().astype(int)
            df_imputed[column] = le_dict[column].inverse_transform(df_imputed[column])
        
        df[categorical_columns_with_missing] = df_imputed
        imputed_columns.extend(categorical_columns_with_missing)
        st.write("Missing values imputed using Random Forest model for categorical columns.")
        st.write(f"Number of rows imputed for categorical columns: {num_imputed_categorical}")
        st.write(f"Columns imputed: {', '.join(categorical_columns_with_missing)}")
        st.write("Top 5 rows after imputation:")
        st.dataframe(df.head())
        
        # Data download option for Random Forest categorical imputation
        if st.button("Download CSV after RF Categorical Imputation"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding for download
            href = f'<a href="data:file/csv;base64,{b64}" download="rf_categorical_imputed_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    if st.checkbox("Fill missing values with Random Forest model for both numeric and categorical columns"):
        num_imputed_numeric += df[numeric_columns_with_missing].isnull().sum().sum()
        num_imputed_categorical += df[categorical_columns_with_missing].isnull().sum().sum()
        imputed_columns.extend(numeric_columns_with_missing)
        imputed_columns.extend(categorical_columns_with_missing)
        # Encoding categorical variables
        le_dict = {}
        for column in categorical_columns_with_missing:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            le_dict[column] = le
        
        # Define imputer for both numeric and categorical columns
        imputer_numeric = IterativeImputer(estimator=RandomForestRegressor(), missing_values=np.nan, max_iter=10, random_state=0)
        df_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(df[numeric_columns_with_missing]), columns=numeric_columns_with_missing)
        
        imputer_categorical = IterativeImputer(estimator=RandomForestRegressor(), missing_values=np.nan, max_iter=10, random_state=0)
        df_categorical_imputed = pd.DataFrame(imputer_categorical.fit_transform(df[categorical_columns_with_missing]), columns=categorical_columns_with_missing)
        
        # Decoding categorical variables
        for column in categorical_columns_with_missing:
            df_categorical_imputed[column] = df_categorical_imputed[column].round().astype(int)
            df_categorical_imputed[column] = le_dict[column].inverse_transform(df_categorical_imputed[column])
        
        df[numeric_columns_with_missing] = df_numeric_imputed
        df[categorical_columns_with_missing] = df_categorical_imputed
        
        st.write("Missing values imputed using Random Forest model for both numeric and categorical columns.")
        st.write(f"Number of rows imputed for numeric columns: {num_imputed_numeric}")
        st.write(f"Number of rows imputed for categorical columns: {num_imputed_categorical}")
        st.write(f"Columns imputed: {', '.join(imputed_columns)}")
        st.write("Top 5 rows after imputation:")
        st.dataframe(df.head())
        
        # Data download option for Random Forest both imputation
        if st.button("Download CSV after RF Both Imputation"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding for download
            href = f'<a href="data:file/csv;base64,{b64}" download="rf_both_imputed_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

    return df

