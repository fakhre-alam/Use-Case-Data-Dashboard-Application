# import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt

# def show_data_profiling(df):
    # st.header("Data Profiling")

    # # Univariate Analysis
    # st.subheader("Univariate Analysis")
    # numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # selected_column = st.selectbox("Select column for univariate analysis", numeric_columns)

    # fig_uni, ax_uni = plt.subplots()
    # sns.histplot(df[selected_column], kde=True, ax=ax_uni)
    # st.pyplot(fig_uni)

    # # Bivariate Analysis
    # st.subheader("Bivariate Analysis")
    # column1 = st.selectbox("Select column 1 for bivariate analysis", numeric_columns, key="column1")
    # column2 = st.selectbox("Select column 2 for bivariate analysis", numeric_columns, key="column2")

    # fig_bi, ax_bi = plt.subplots()
    # sns.scatterplot(x=df[column1], y=df[column2], ax=ax_bi)
    # st.pyplot(fig_bi)

# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# def show_data_profiling(df):
    # st.header("Data Profiling")

    # # Univariate Analysis
    # st.subheader("Univariate Analysis")
    # numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # selected_column = st.selectbox("Select column for univariate analysis", numeric_columns)

    # fig_uni, ax_uni = plt.subplots()
    # sns.histplot(df[selected_column], kde=True, ax=ax_uni)
    # st.pyplot(fig_uni)

    # # Bivariate Analysis
    # st.subheader("Bivariate Analysis")
    # column1 = st.selectbox("Select column 1 for bivariate analysis", numeric_columns, key="column1")
    # column2 = st.selectbox("Select column 2 for bivariate analysis", numeric_columns, key="column2")

    # fig_bi, ax_bi = plt.subplots()
    # sns.scatterplot(x=df[column1], y=df[column2], ax=ax_bi)
    # st.pyplot(fig_bi)

    # # Correlation Matrix
    # st.subheader("Correlation Matrix")
    # columns = df.columns.tolist()
    # selected_columns_corr = st.multiselect("Select variables for correlation matrix", numeric_columns)

    # if selected_columns_corr:
        # corr_df = df[selected_columns_corr].corr()

        # # Displaying correlation matrix
        # st.write("Correlation Matrix:")
        # st.write(corr_df)

        # # Plotting the correlation matrix
        # st.write("Correlation Heatmap:")
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        # st.pyplot()

    # # Multivariate Analysis
    # st.subheader("Multivariate Analysis")
    # pairplot_columns = st.multiselect("Select variables for pairplot", numeric_columns)

    # if pairplot_columns:
        # st.write("Pairplot:")
        # sns.pairplot(df[pairplot_columns])
        # st.pyplot()
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# def show_data_profiling(df):
    # st.header("Data Profiling")

    # # Univariate Analysis
    # st.subheader("Univariate Analysis")
    # numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # selected_column = st.selectbox("Select column for univariate analysis", numeric_columns)

    # if selected_column:
        # fig_uni, ax_uni = plt.subplots()
        # sns.histplot(df[selected_column], kde=True, ax=ax_uni)
        # ax_uni.set_title(f'Univariate Analysis of {selected_column}')
        # st.pyplot(fig_uni)

    # # Bivariate Analysis
    # st.subheader("Bivariate Analysis")
    # column1 = st.selectbox("Select column 1 for bivariate analysis", numeric_columns, key="column1")
    # column2 = st.selectbox("Select column 2 for bivariate analysis", numeric_columns, key="column2")

    # if column1 and column2:
        # fig_bi, ax_bi = plt.subplots()
        # sns.scatterplot(x=df[column1], y=df[column2], ax=ax_bi)
        # ax_bi.set_title(f'Bivariate Analysis between {column1} and {column2}')
        # st.pyplot(fig_bi)

    # # Correlation Matrix
    # st.subheader("Correlation Matrix")
    # selected_columns_corr = st.multiselect("Select variables for correlation matrix", numeric_columns)

    # if selected_columns_corr:
        # corr_df = df[selected_columns_corr].corr()

        # # Displaying correlation matrix
        # st.write("Correlation Matrix:")
        # st.dataframe(corr_df)

        # # Plotting the correlation matrix
        # st.write("Correlation Heatmap:")
        # fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        # sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax_corr)
        # ax_corr.set_title('Correlation Heatmap')
        # st.pyplot(fig_corr)

    # # Multivariate Analysis
    # st.subheader("Multivariate Analysis")
    # pairplot_columns = st.multiselect("Select variables for pairplot", numeric_columns)

    # if pairplot_columns:
        # st.write("Pairplot:")
        # pairplot_fig = sns.pairplot(df[pairplot_columns])
        # st.pyplot(pairplot_fig)
        
        


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_data_profiling(df):
    st.header("Data Profiling")

    # Univariate Analysis
    st.subheader("Univariate Analysis")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_column = st.selectbox("Select column for univariate analysis", numeric_columns)

    if selected_column:
        fig_uni, ax_uni = plt.subplots()
        sns.histplot(df[selected_column], kde=True, ax=ax_uni)
        ax_uni.set_title(f'Univariate Analysis of {selected_column}')
        st.pyplot(fig_uni)

    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    column1 = st.selectbox("Select column 1 for bivariate analysis", numeric_columns + df.select_dtypes(include=['object']).columns.tolist(), key="column1")
    column2 = st.selectbox("Select column 2 for bivariate analysis", numeric_columns + df.select_dtypes(include=['object']).columns.tolist(), key="column2")

    if column1 and column2:
        fig_bi, ax_bi = plt.subplots()
        if df[column1].dtype == 'object' or df[column2].dtype == 'object':
            sns.scatterplot(x=df[column1], y=df[column2], ax=ax_bi, hue=df[column1])
        else:
            sns.scatterplot(x=df[column1], y=df[column2], ax=ax_bi)
        ax_bi.set_title(f'Bivariate Analysis between {column1} and {column2}')
        st.pyplot(fig_bi)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    selected_columns_corr = st.multiselect("Select variables for correlation matrix", numeric_columns)

    if selected_columns_corr:
        corr_df = df[selected_columns_corr].corr()

        # Displaying correlation matrix
        st.write("Correlation Matrix:")
        st.dataframe(corr_df)

        # Plotting the correlation matrix
        st.write("Correlation Heatmap:")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax_corr)
        ax_corr.set_title('Correlation Heatmap')
        st.pyplot(fig_corr)

    # Multivariate Analysis
    st.subheader("Multivariate Analysis")
    pairplot_columns = st.multiselect("Select variables for pairplot", numeric_columns)

    if pairplot_columns:
        st.write("Pairplot:")
        pairplot_fig = sns.pairplot(df[pairplot_columns])
        st.pyplot(pairplot_fig)


