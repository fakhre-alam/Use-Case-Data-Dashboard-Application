import streamlit as st
import io
import numpy as np
import pandas as pd
def show_about_data(df):
    st.header("About Data")
    
    st.subheader("Data Shape")
    st.write(df.shape)
    
    st.subheader("Data Head")
    st.write(df.head())

    st.subheader("Data Tail")
    st.write(df.tail())

    st.subheader("Data Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Column Counts")
    col_count_data = {
        "Column Type": ["Numerical Columns", "Categorical Columns", "Date Columns"],
        "Count": [
            len(df.select_dtypes(include=[np.number]).columns.tolist()),
            len(df.select_dtypes(include=['object']).columns.tolist()),
            len(df.select_dtypes(include=['datetime']).columns.tolist())
        ]
    }
    col_count_df = pd.DataFrame(col_count_data)
    st.table(col_count_df)
