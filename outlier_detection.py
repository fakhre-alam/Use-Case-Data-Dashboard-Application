# import streamlit as st
# import pandas as pd
# from sklearn.ensemble import IsolationForest
# import matplotlib.pyplot as plt

# def detect_outliers(df):
    # st.header("Outlier Detection")

    # numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # selected_column = st.selectbox("Select column for outlier detection", numeric_columns)

    # iso = IsolationForest(contamination=0.1)
    # df['anomaly'] = iso.fit_predict(df[[selected_column]])

    # outliers = df[df['anomaly'] == -1]
    # st.write("Outliers detected:", len(outliers))

    # plt.figure(figsize=(10, 6))
    # plt.scatter(df.index, df[selected_column], c=df['anomaly'], cmap='coolwarm')
    # st.pyplot()
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import IsolationForest
# import matplotlib.pyplot as plt
# import seaborn as sns

# def detect_outliers(df):
    # st.header("Outlier Detection")

    # # Select numeric columns for outlier detection
    # numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    # selected_column = st.selectbox("Select column for outlier detection", numeric_columns)

    # # Contamination slider
    # contamination = st.slider("Select contamination level", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

    # # Isolation Forest for outlier detection
    # isolation_forest = IsolationForest(contamination=contamination)
    # df['anomaly_score'] = isolation_forest.fit_predict(df[[selected_column]])
    # df['anomaly'] = isolation_forest.fit_predict(df[[selected_column]])

    # # Calculate the anomaly scores
    # df['anomaly_score'] = isolation_forest.decision_function(df[[selected_column]])
    # outliers1 = df[df['anomaly'] == -1]
    # st.write("Outliers detected:", len(outliers1))
    # outliers = df[df['anomaly_score'] < 0].sort_values(by='anomaly_score').head(5)

    # st.subheader("Top 5 Outliers")
    # st.dataframe(outliers)

    # # Plotting the anomaly scores
    # fig, ax = plt.subplots()
    # sns.histplot(df['anomaly_score'], bins=50, kde=True, ax=ax)
    # ax.set_title('Anomaly Score Distribution')
    # st.pyplot(fig)

    # # Scatter plot of the outliers
    # fig, ax = plt.subplots()
    # sns.scatterplot(x=df.index, y=selected_column, hue='anomaly_score', data=df, ax=ax, palette='coolwarm', legend=False)
    # ax.set_title(f'Outliers in {selected_column}')
    # st.pyplot(fig)

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import IsolationForest
# import matplotlib.pyplot as plt
# import seaborn as sns

# def detect_outliers(df):
    # st.header("Outlier Detection")

    # # Select numeric columns for outlier detection
    # numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    # selected_column = st.selectbox("Select column for outlier detection", numeric_columns)

    # # Contamination slider
    # contamination = st.slider("Select contamination level", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

    # # Isolation Forest for outlier detection
    # isolation_forest = IsolationForest(contamination=contamination)
    # df['anomaly_score'] = isolation_forest.fit_predict(df[[selected_column]])
    # df['anomaly'] = isolation_forest.fit_predict(df[[selected_column]])
    # # Calculate the anomaly scores
    # df['anomaly_score'] = isolation_forest.decision_function(df[[selected_column]])
    # outliers1 = df[df['anomaly'] == -1]
    # st.write("Outliers detected:", len(outliers1))
    # outliers = df[df['anomaly_score'] < 0].sort_values(by='anomaly_score').head(5)
    
    # st.subheader("Top 5 Outliers")
    # st.dataframe(outliers)

    # # Plotting the anomaly scores
    # fig, ax = plt.subplots()
    # sns.histplot(df['anomaly_score'], bins=50, kde=True, ax=ax)
    # ax.set_title('Anomaly Score Distribution')
    # st.pyplot(fig)

    # # Scatter plot of the outliers
    # df['is_outlier'] = df['anomaly_score'] < 0
    # df.reset_index(drop=True, inplace=True)  # Resetting index
    # fig, ax = plt.subplots()
    # sns.scatterplot(x=df.index, y=selected_column, hue='is_outlier', palette={True: 'red', False: 'blue'}, data=df, ax=ax)
    # ax.set_title(f'Outliers in {selected_column}')
    # handles, labels = ax.get_legend_handles_labels()
    # labels = ['Normal', 'Outlier']
    # ax.legend(handles=handles, labels=labels, title='Legend')
    # st.pyplot(fig)




# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import IsolationForest
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.impute import SimpleImputer

# def detect_outliers(df):
    # st.header("Outlier Detection")

    # # Select numeric columns for outlier detection
    # numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    # selected_column = st.selectbox("Select column for outlier detection", numeric_columns)

    # # Contamination slider
    # contamination = st.slider("Select contamination level", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

    # # Handle missing values
    # missing_option = st.radio("How do you want to handle missing values?", ("Drop rows with missing values", "Impute missing values"))
    
    # if missing_option == "Drop rows with missing values":
        # df = df.dropna(subset=[selected_column])
    # elif missing_option == "Impute missing values":
        # imputer = SimpleImputer(strategy='mean')
        # df[selected_column] = imputer.fit_transform(df[[selected_column]])

    # if df[selected_column].isnull().sum() > 0:
        # st.warning(f"There are still missing values in the column {selected_column}. Please handle them.")
        # return

    # # Isolation Forest for outlier detection
    # isolation_forest = IsolationForest(contamination=contamination)
    # df['anomaly'] = isolation_forest.fit_predict(df[[selected_column]])
    # df['anomaly_score'] = isolation_forest.decision_function(df[[selected_column]])
    
    # outliers = df[df['anomaly'] == -1]
    # st.write("Outliers detected:", len(outliers))
    
    # st.subheader("Top 5 Outliers")
    # top_outliers = outliers.sort_values(by='anomaly_score').head(5)
    # st.dataframe(top_outliers)

    # # Plotting the anomaly scores
    # fig, ax = plt.subplots()
    # sns.histplot(df['anomaly_score'], bins=50, kde=True, ax=ax)
    # ax.set_title('Anomaly Score Distribution')
    # st.pyplot(fig)

    # # Scatter plot of the outliers
    # df['is_outlier'] = df['anomaly'] == -1
    # df.reset_index(drop=True, inplace=True)  # Resetting index
    # fig, ax = plt.subplots()
    # sns.scatterplot(x=df.index, y=selected_column, hue='is_outlier', palette={True: 'red', False: 'blue'}, data=df, ax=ax)
    # ax.set_title(f'Outliers in {selected_column}')
    # handles, labels = ax.get_legend_handles_labels()
    # labels = ['Normal', 'Outlier']
    # ax.legend(handles=handles, labels=labels, title='Legend')
    # st.pyplot(fig)
    
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

def detect_outliers(df):
    st.header("Outlier Detection")

    # Select numeric columns for outlier detection
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_column = st.selectbox("Select column for outlier detection", numeric_columns)

    # Contamination slider
    contamination = st.slider("Select contamination level", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

    # Handle missing values
    missing_option = st.radio("How do you want to handle missing values?", ("Drop rows with missing values", "Impute missing values"))
    
    if missing_option == "Drop rows with missing values":
        df = df.dropna(subset=[selected_column])
    elif missing_option == "Impute missing values":
        imputer = SimpleImputer(strategy='mean')
        df[selected_column] = imputer.fit_transform(df[[selected_column]])

    if df[selected_column].isnull().sum() > 0:
        st.warning(f"There are still missing values in the column {selected_column}. Please handle them.")
        return

    # Isolation Forest for outlier detection
    isolation_forest = IsolationForest(contamination=contamination)
    df['anomaly'] = isolation_forest.fit_predict(df[[selected_column]])
    df['anomaly_score'] = isolation_forest.decision_function(df[[selected_column]])
    
    outliers = df[df['anomaly'] == -1]
    st.write("Outliers detected:", len(outliers))
    
    st.subheader("Top 5 Outliers")
    top_outliers = outliers.sort_values(by='anomaly_score').head(5)
    st.dataframe(top_outliers)

    # Plotting the anomaly scores
    fig, ax = plt.subplots()
    sns.histplot(df['anomaly_score'], bins=50, kde=True, ax=ax)
    ax.set_title('Anomaly Score Distribution')
    st.pyplot(fig)

    # Scatter plot of the outliers
    df['is_outlier'] = df['anomaly'] == -1
    df.reset_index(drop=True, inplace=True)  # Resetting index
    fig, ax = plt.subplots()
    sns.scatterplot(x=df.index, y=selected_column, hue='is_outlier', palette={True: 'red', False: 'blue'}, data=df, ax=ax)
    ax.set_title(f'Outliers in {selected_column}')
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Normal', 'Outlier']
    ax.legend(handles=handles, labels=labels, title='Legend')
    st.pyplot(fig)

    # Download button for outliers
    st.subheader("Download Outliers")
    csv = outliers.to_csv(index=False)
    st.download_button(
        label="Download Outliers as CSV",
        data=csv,
        file_name='outliers.csv',
        mime='text/csv'
    )

