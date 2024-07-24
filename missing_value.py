# import streamlit as st
# import pandas as pd

# def handle_missing_values(df):
    # st.header("Missing Value Identification and Handling")

    # st.subheader("Missing Values")
    # st.write(df.isnull().sum())

    # action = st.selectbox("Select Action for Missing Values", ("Drop missing values", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill Categorical with 'Other'"))

    # if st.button("Apply"):
        # if action == "Drop missing values":
            # df = df.dropna()
        # elif action == "Fill with Mean":
            # df = df.fillna(df.mean())
        # elif action == "Fill with Median":
            # df = df.fillna(df.median())
        # elif action == "Fill with Mode":
            # df = df.fillna(df.mode().iloc[0])
        # elif action == "Fill Categorical with 'Other'":
            # df = df.fillna("Other")
        
        # st.write("Missing values handled")
        # st.write(df.isnull().sum())
        
    # return df


import streamlit as st
import pandas as pd
import base64  # Needed for encoding the CSV file for download

def handle_missing_values(df):
    st.header("Missing Value Identification and Handling")

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    action = st.selectbox("Select Action for Missing Values", ("Drop missing values", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill Categorical with 'Other'"))

    if st.button("Apply"):
        if action == "Drop missing values":
            df = df.dropna()
        elif action == "Fill with Mean":
            df = df.fillna(df.mean())
        elif action == "Fill with Median":
            df = df.fillna(df.median())
        elif action == "Fill with Mode":
            df = df.fillna(df.mode().iloc[0])
        elif action == "Fill Categorical with 'Other'":
            df = df.fillna("Other")
        
        st.write("Missing values handled")
        st.write(df.isnull().sum())
        
        # Data download option
        if not df.empty:
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding for download
            href = f'<a href="data:file/csv;base64,{b64}" download="cleaned_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

    return df

