# import streamlit as st
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier, XGBRegressor
# from sklearn.metrics import accuracy_score, mean_squared_error
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# import pandas as pd

# def train_model(df):
    # st.header("Model Training")

    # # Select target column
    # target_column = st.selectbox("Select Target Column", df.columns)

    # # Select model type: Classification or Regression
    # model_type = st.selectbox("Select Model Type", ("Classification", "Regression"))

    # # Define features and target
    # X = df.drop(columns=[target_column])
    # y = df[target_column]
    
    # # Encode target if classification
    # if model_type == 'Classification':
        # if y.dtype == 'object':
            # le = LabelEncoder()
            # y = le.fit_transform(y)

    # # Split data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # if model_type == 'Classification':
        # model = st.selectbox("Select Model", ("Random Forest", "Logistic Regression", "XGBoost"))
        # if model == "Random Forest":
            # clf = RandomForestClassifier()
        # elif model == "Logistic Regression":
            # clf = LogisticRegression()
        # elif model == "XGBoost":
            # clf = XGBClassifier()
        
        # clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # st.write("Accuracy:", accuracy_score(y_test, y_pred))

        # if model != "Logistic Regression":
            # feature_importances = clf.feature_importances_
            # fig, ax = plt.subplots()
            # pd.Series(feature_importances, index=X.columns).nlargest(10).plot(kind='barh', ax=ax)
            # ax.set_title('Feature Importances')
            # st.pyplot(fig)

    # elif model_type == 'Regression':
        # model = st.selectbox("Select Model", ("Random Forest", "XGBoost"))
        # if model == "Random Forest":
            # reg = RandomForestRegressor()
        # elif model == "XGBoost":
            # reg = XGBRegressor()

        # reg.fit(X_train, y_train)
        # y_pred = reg.predict(X_test)
        # st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))

        # feature_importances = reg.feature_importances_
        # fig, ax = plt.subplots()
        # pd.Series(feature_importances, index=X.columns).nlargest(10).plot(kind='barh', ax=ax)
        # ax.set_title('Feature Importances')
        # st.pyplot(fig)

# import streamlit as st
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier, XGBRegressor
# from sklearn.metrics import accuracy_score, mean_squared_error
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# import pandas as pd

# def train_model(df):
    # st.header("Model Training")

    # # Select target column
    # target_column = st.selectbox("Select Target Column", df.columns)

    # # Select model type: Classification or Regression
    # model_type = st.selectbox("Select Model Type", ("Classification", "Regression"))

    # # Define features and target
    # X = df.drop(columns=[target_column])
    # y = df[target_column]

    # # Encode target if classification
    # if model_type == 'Classification':
        # if y.dtype == 'object':
            # le = LabelEncoder()
            # y = le.fit_transform(y)

    # # Apply label encoding to feature columns
    # for column in X.columns:
        # if X[column].dtype == 'object':
            # le = LabelEncoder()
            # X[column] = le.fit_transform(X[column])

    # # Split data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # if model_type == 'Classification':
        # model = st.selectbox("Select Model", ("Random Forest", "Logistic Regression", "XGBoost"))
        # if model == "Random Forest":
            # clf = RandomForestClassifier()
        # elif model == "Logistic Regression":
            # clf = LogisticRegression()
        # elif model == "XGBoost":
            # clf = XGBClassifier()

        # clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # st.write("Accuracy:", accuracy_score(y_test, y_pred))

        # if model != "Logistic Regression":
            # feature_importances = clf.feature_importances_
            # fig, ax = plt.subplots()
            # pd.Series(feature_importances, index=X.columns).nlargest(10).plot(kind='barh', ax=ax)
            # ax.set_title('Feature Importances')
            # st.pyplot(fig)

    # elif model_type == 'Regression':
        # model = st.selectbox("Select Model", ("Random Forest", "XGBoost"))
        # if model == "Random Forest":
            # reg = RandomForestRegressor()
        # elif model == "XGBoost":
            # reg = XGBRegressor()

        # reg.fit(X_train, y_train)
        # y_pred = reg.predict(X_test)
        # st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))

        # feature_importances = reg.feature_importances_
        # fig, ax = plt.subplots()
        # pd.Series(feature_importances, index=X.columns).nlargest(10).plot(kind='barh', ax=ax)
        # ax.set_title('Feature Importances')
        # st.pyplot(fig)





# import streamlit as st
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier, XGBRegressor
# from sklearn.metrics import accuracy_score, mean_squared_error
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# import pandas as pd

# def train_model(df):
    # st.header("Model Training")

    # # Select target column
    # target_column = st.selectbox("Select Target Column", df.columns)

    # # Select independent variables
    # feature_columns = st.multiselect("Select Independent Variables", df.columns.drop(target_column))

    # if not feature_columns:
        # st.warning("Please select at least one independent variable.")
        # return

    # # Select model type: Classification or Regression
    # model_type = st.selectbox("Select Model Type", ("Classification", "Regression"))

    # # Define features and target
    # X = df[feature_columns]
    # y = df[target_column]

    # # Encode target if classification
    # if model_type == 'Classification':
        # if y.dtype == 'object':
            # le = LabelEncoder()
            # y = le.fit_transform(y)

    # # Apply label encoding to feature columns
    # for column in X.columns:
        # if X[column].dtype == 'object':
            # le = LabelEncoder()
            # X[column] = le.fit_transform(X[column])

    # # Handle missing values
    # if st.checkbox("Fill missing values"):
        # for column in X.columns:
            # if X[column].dtype == 'object':
                # X[column] = X[column].fillna(X[column].mode()[0])
            # else:
                # X[column] = X[column].fillna(X[column].mean())

    # # Split data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # if model_type == 'Classification':
        # model = st.selectbox("Select Model", ("Random Forest", "Logistic Regression", "XGBoost"))
        # if model == "Random Forest":
            # clf = RandomForestClassifier()
        # elif model == "Logistic Regression":
            # clf = LogisticRegression()
        # elif model == "XGBoost":
            # clf = XGBClassifier()

        # clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # st.write("Accuracy:", accuracy_score(y_test, y_pred))

        # if model != "Logistic Regression":
            # feature_importances = clf.feature_importances_
            # fig, ax = plt.subplots()
            # pd.Series(feature_importances, index=X.columns).nlargest(10).plot(kind='barh', ax=ax)
            # ax.set_title('Feature Importances')
            # st.pyplot(fig)

    # elif model_type == 'Regression':
        # model = st.selectbox("Select Model", ("Random Forest", "XGBoost"))
        # if model == "Random Forest":
            # reg = RandomForestRegressor()
        # elif model == "XGBoost":
            # reg = XGBRegressor()

        # reg.fit(X_train, y_train)
        # y_pred = reg.predict(X_test)
        # st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))

        # feature_importances = reg.feature_importances_
        # fig, ax = plt.subplots()
        # pd.Series(feature_importances, index=X.columns).nlargest(10).plot(kind='barh', ax=ax)
        # ax.set_title('Feature Importances')
        # st.pyplot(fig)


import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def train_model(df):
    st.header("Model Training")

    # Select target column
    target_column = st.selectbox("Select Target Column", df.columns)

    # Select independent variables
    feature_columns = st.multiselect("Select Independent Variables", df.columns.drop(target_column))

    if not feature_columns:
        st.warning("Please select at least one independent variable.")
        return

    # Select model type: Classification or Regression
    model_type = st.selectbox("Select Model Type", ("Classification", "Regression"))

    # Define features and target
    X = df[feature_columns]
    y = df[target_column]

    # Encode target if classification
    le = None
    if model_type == 'Classification':
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

    # Apply label encoding to feature columns
    for column in X.columns:
        if X[column].dtype == 'object':
            le_feature = LabelEncoder()
            X[column] = le_feature.fit_transform(X[column])

    # Handle missing values
    if st.checkbox("Fill missing values"):
        for column in X.columns:
            if X[column].dtype == 'object':
                X[column] = X[column].fillna(X[column].mode()[0])
            else:
                X[column] = X[column].fillna(X[column].mean())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if model_type == 'Classification':
        model = st.selectbox("Select Model", ("Random Forest", "Logistic Regression", "XGBoost"))
        if model == "Random Forest":
            clf = RandomForestClassifier()
        elif model == "Logistic Regression":
            clf = LogisticRegression()
        elif model == "XGBoost":
            clf = XGBClassifier()

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cmd = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots()
        cmd.plot(ax=ax)
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        st.write("The confusion matrix shows the number of correct and incorrect predictions made by the model, categorized by actual and predicted classes.")

        # Classification Report
        if le is not None:
            target_names = le.classes_
        else:
            target_names = np.unique(y_test)
        class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(class_report).transpose())
        st.write("The classification report provides precision, recall, f1-score, and support for each class, which help in evaluating the model's performance.")

        if model != "Logistic Regression":
            feature_importances = clf.feature_importances_
            fig, ax = plt.subplots()
            pd.Series(feature_importances, index=X.columns).nlargest(10).plot(kind='barh', ax=ax)
            ax.set_title('Feature Importances')
            st.pyplot(fig)

    elif model_type == 'Regression':
        model = st.selectbox("Select Model", ("Random Forest", "XGBoost"))
        if model == "Random Forest":
            reg = RandomForestRegressor()
        elif model == "XGBoost":
            reg = XGBRegressor()

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write("Mean Squared Error:", mse)

        feature_importances = reg.feature_importances_
        fig, ax = plt.subplots()
        pd.Series(feature_importances, index=X.columns).nlargest(10).plot(kind='barh', ax=ax)
        ax.set_title('Feature Importances')
        st.pyplot(fig)
