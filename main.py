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

        header_option = st.selectbox("Does the file have a header?", ["Yes", "No"])
        if header_option == "Yes":
            header_row = st.number_input("Header row", min_value=0, value=0)
        else:
            header_row = None

        delimiter = st.text_input("Delimiter", value=",")

        custom_headers = st.text_area("Custom feature names (comma-separated)", "")

        try:
            if header_row==None:
                df = pd.read_csv(uploaded_file, header=None, delimiter=delimiter)
                df.columns = [f"Column {i}" for i in range(len(df.columns))]
            else:
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

    st.write(df)

    nbRows = df.shape[0]
    st.write("There is " + str(nbRows) + " rows")

    nbColumns = df.shape[1]
    st.write("There is " + str(nbColumns) + " columns")

    NaN = df.isna().sum().sum()
    st.write("There is " + str(NaN) + " missing values")


if uploaded_file is not None:
    st.subheader('Data Visualizations')

    import plotly.express as px

    selected_column = st.selectbox("Select a column to visualize", df.columns)

    if selected_column:

        # Histogram
        fig_histogram = px.histogram(df, x=selected_column, title=f'Histogram of {selected_column}', nbins=30)
        st.plotly_chart(fig_histogram)

        # Box Plot
        fig_boxplot = px.box(df, y=selected_column, title=f'Box Plot of {selected_column}')
        st.plotly_chart(fig_boxplot)

        # Bar Plot
        value_counts = df[selected_column].value_counts().reset_index()
        value_counts.columns = [selected_column, 'count']
        fig_barplot = px.bar(value_counts, x=selected_column, y='count', title=f'Bar Plot of {selected_column}')
        st.plotly_chart(fig_barplot)

        # Violin plot
        fig_violin = px.violin(df, y=selected_column, title=f'Violin Plot of {selected_column}')
        st.plotly_chart(fig_violin)

        # Pie chart
        fig_pie = px.pie(value_counts, names=selected_column, values='count', title=f'Pie Chart of {selected_column}')
        st.plotly_chart(fig_pie)
        
        if (len(df.columns)>=2):
            columns = st.multiselect("Select columns for Scatter Plot", df.columns, max_selections=2)
            if len(columns) >= 2:
                fig_scatter = px.scatter(df, x=columns[0], y=columns[1],
                                            title=f'Scatter Plot: {columns[0]} vs {columns[1]}')
                st.plotly_chart(fig_scatter)

        st.markdown("---")
