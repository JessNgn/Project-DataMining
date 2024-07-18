import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import numpy as np

st.markdown("Tutor: Issam FALIH")
st.markdown("BIA2 - Nhat-Vy Jessica NGUYEN - Kevin OP")

st.title("Data Mining - Project")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:

    try:
        
        header_option = st.selectbox("Does the file have a header?", ["Yes", "No"])

        if header_option == "Yes":

            header_row = st.number_input("Header row", min_value=0, value=0)

        else:

            header_row = None

        delimiter = st.text_input("Delimiter", value=",")

        custom_headers = st.text_area("Custom feature names (comma-separated)", "").strip()
        custom_headers_list = [x.strip() for x in custom_headers.split(',')] if custom_headers else None

        encoding = st.selectbox("File Encoding", ['utf-8', 'latin1', 'iso-8859-1'])

        if header_option == "Yes":

            df = pd.read_csv(uploaded_file, header=header_row, delimiter=delimiter, encoding=encoding)

        else:
            
            df = pd.read_csv(uploaded_file, header=None, delimiter=delimiter, encoding=encoding)
        
        if custom_headers_list:

            if len(custom_headers_list) == len(df.columns):

                df.columns = custom_headers_list

            else:
                
                st.warning("The number of custom feature names does not match the number of columns in the CSV.")
        
        st.success("File loaded successfully!")

        st.write("")
        st.write("Let's check if the dataset has been loaded successfully")

        nbRows = df.shape[0]
        st.write("There is " + str(nbRows) + " rows")

        nbColumns = df.shape[1]
        st.write("There is " + str(nbColumns) + " columns")

        st.write(df)

        if (not df.isnull().any(axis=0).any()) and (not df.isnull().any(axis=1).any()):
            
            st.subheader("There is not a single missing value inside your data.")

        else:

            st.subheader("There are some missing values inside your data.")

            NaN_choice = st.selectbox("Do you want to delete the empty rows and columns ?", ["No", "Yes"])

            if NaN_choice == "Yes":

                df = df.dropna(how='all', axis=0)
                df = df.dropna(how='all', axis=1)

                st.write(df)

                NaN_option = st.selectbox("Do you want to delete some columns ?", ["No", "Yes"])

                if NaN_option == "Yes":

                    df2 = pd.DataFrame({
                        "Variable": df.columns,
                        "Number of Missing values": [df[i].isna().sum() for i in df.columns],
                        "Percentage of Missing values": [df[i].isna().sum() / len(df) * 100 for i in df.columns]
                    })

                    NaN_threshold = st.number_input("Level of thresold for missing values ?")
                    df2 = df2[df2["Percentage of Missing values"] <= NaN_threshold] 
                    st.table(df2)

                    NaN_replace = st.selectbox("Do you want to replace the missing values ?", ["No", "Yes"])

                    if NaN_replace == "Yes":

                        for i in df.columns:

                            if df[i].isnull().any():

                                st.subheader(f"{i}")

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write(f"Type: {df[i].dtype}")

                                with col2:

                                    if df[i].dtype != "object":                                    

                                        Replace_method = st.selectbox(f"Which method do you want to use to replace the missing values inside {i} ?", ["None", "Mean", "Median", "Mode"])

                                        if Replace_method == "None":
                                            continue

                                        if Replace_method == "Mean":
                                            imputer = SimpleImputer(strategy='mean')

                                        if Replace_method == "Median":
                                            imputer = SimpleImputer(strategy='median')

                                        if Replace_method == "Mode":
                                            imputer = SimpleImputer(strategy='most_frequent')

                                        df[[i]] = imputer.fit_transform(df[[i]])
                                    
                                    else:

                                        Replace_method = st.selectbox(f"Which method do you want to use to replace the missing values inside {i} ?", ["None", "Mode"])
                                        
                                        if Replace_method == "None":
                                            continue                        

                                        if Replace_method == "Mode":
                                            imputer = SimpleImputer(strategy='most_frequent')
                                        
                                        df[[i]] = imputer.fit_transform(df[[i]])

                            else:

                                st.write(f"There is no missing value inside {i}")
                        
                        st.table(df)

                else:

                    df2 = pd.DataFrame({
                        "Variable": df.columns,
                        "Number of Missing values": [df[i].isna().sum() for i in df.columns],
                        "Percentage of Missing values": [df[i].isna().sum() / len(df) * 100 for i in df.columns]
                    })

                    st.table(df2)
            
            else:
                df2 = pd.DataFrame({
                    "Variable": df.columns,
                    "Number of Missing values": [df[i].isna().sum() for i in df.columns],
                    "Percentage of Missing values": [df[i].isna().sum() / len(df) * 100 for i in df.columns]
                })
                    
                st.table(df2)

        Normal_choice = st.selectbox("Do you want to normalize your data ?", ["No", "Yes"])

        df_norm = df

        if Normal_choice == "Yes":

            Normal_method = st.selectbox(f"Which method do you want to use ?", ["Normalize", "Min-Max", "Z-score", "Log Scaling"])
            features = st.multiselect("Which feature do you want to select ?", df.select_dtypes(include=['number']).columns)

            if not features: 

                st.write("You didn't select any feature yet.")
            
            else:

                if Normal_method == "Min-Max":
                    scaler = preprocessing.MinMaxScaler()
                    df_norm[features] = scaler.fit_transform(df_norm[features])

                if Normal_method == "Z-score":
                    scaler = preprocessing.StandardScaler()
                    df_norm[features] = scaler.fit_transform(df_norm[features])

                if Normal_method == "Normalize":
                    df_norm[features] = preprocessing.normalize(df_norm[features])    

                if Normal_method == "Log Scaling":
                    df_norm[features] = np.log(df_norm[features]) 

            st.write(df_norm)

        else:

            st.write(df_norm)

    except Exception as e:
        st.error(f"Error loading CSV file: Try to change the File Encoding : str{e}")
else:
    st.write("Please upload a CSV file.")