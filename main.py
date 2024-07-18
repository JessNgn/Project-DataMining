import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.markdown("Tutor: Issam FALIH")
st.markdown("BIA2 - Nhat-Vy Jessica NGUYEN - Kevin OP")

st.title("Data Mining - Project")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:

    if uploaded_file.name.endswith('.csv'):

        header_row = st.number_input("Header row (0-indexed)", min_value=0, value=0)
        delimiter = st.text_input("Delimiter", value=",")

        custom_headers = st.text_area("Custom feature names (comma-separated)", "")

        try:
            df = pd.read_csv(uploaded_file, header=header_row, delimiter=delimiter)
            
            if custom_headers:
                custom_headers_list = [x.strip() for x in custom_headers.split(',')]
                if len(custom_headers_list) == len(df.columns):
                    df.columns = custom_headers_list
                else:
                    st.warning("The number of custom feature names does not match the number of columns in the CSV.")

        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
    else:
        st.error("The uploaded file is not a CSV. Please upload a CSV file.")
else:
    st.write("Please upload a CSV file.")


if uploaded_file is not None:
    st.write("")
    st.write("Let's check if the dataset has been loaded successfully")
    
    st.write(df.head())
    st.write(df.tail())

    nbRows = df.shape[0]
    st.write("There is " + str(nbRows) + " rows")

    nbColumns = df.shape[1]
    st.write("There is " + str(nbColumns) + " columns")

    NaN = df.isna().sum().sum()
    st.write("There is " + str(NaN) + " missing values")