#! /bin/python
# -*- coding: utf-8 -*-
#
# main.py
#

__authors__ = "Nhat-Vy Jessica NGUYEN - Kévin OP"
__copyright__ = "Copyright 2024, DataMiningProject 2024"
__credits__ = ["Nhat-Vy Jessica NGUYEN - Kévin OP"]
__version__ = "0.0.1"
__maintainers__ = "Nhat-Vy Jessica NGUYEN -  Kévin OP"
__emails__ = "nhat-vy-jessica.nguyen@efrei.net - kevin.op@efrei.net"
__status__ = "Research code"

# ----------------------------------------------- Librairies  ---------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report



# ---------------------------------- Side Bar / Presentattion -------------------------------------

st.sidebar.title("Data Mining - Project")

st.sidebar.info("Nhat-Vy Jessica NGUYEN - Kevin OP          BIA2")
st.sidebar.info("Tutor: Issam FALIH")

st.sidebar.markdown("---")

st.sidebar.markdown('<a href="mailto:nhat-vy-jessica.nguyen@efrei.net" style="color: #ADD8E6; text-decoration: none;">:e-mail: nhat-vy-jessica.nguyen@efrei.net</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="mailto:kevin.op@efrei.net" style="color: #ADD8E6; text-decoration: none;">:e-mail: kevin.op@efrei.net</a>', unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown('<a href="https://github.com/JessNgn/Project-DataMining" style="color: #ADD8E6; text-decoration: none;">:link: GitHub</a>', unsafe_allow_html=True)



# -------------------------------------------- Pages ----------------------------------------------

tab, tab1, tab2= st.tabs(["Data", "Visualizations", "Clustering & Predictions"])

with tab:

    st.title("Data Mining - Project")

    uploaded_file = st.file_uploader("Choose a CSV file")

    if uploaded_file is not None:

        try:
            
            header_option = st.selectbox("Does the file have a header?", ["Yes", "No"])

            if header_option == "Yes":
                header_row = st.number_input("Header row", min_value=0, value=0)
            else:
                header_row = None
        
            if header_option == "No":
                header_row = None

            delimiter = st.text_input("Delimiter", value=",")

            custom_headers = st.text_area("Custom feature names (comma-separated)", "").strip()
            custom_headers_list = [x.strip() for x in custom_headers.split(',')] if custom_headers else None

            encoding = st.selectbox("File Encoding", ['utf-8', 'latin1', 'iso-8859-1'])

            if header_option == "Yes":

                df = pd.read_csv(uploaded_file, header=header_row, delimiter=delimiter, encoding=encoding)
            else:
                df = pd.read_csv(uploaded_file, header=None, delimiter=delimiter, encoding=encoding)
                df.columns = [f"Column {i}" for i in range(len(df.columns))]
            
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

                NaN_choice = st.selectbox("Do you want to take care of the missing values ?", ["No", "Yes"])

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

                        NaN_threshold = st.number_input("Level of threshold for missing values ?")
                        df2 = df2[df2["Percentage of Missing values"] <= NaN_threshold] 
                        st.table(df2)

                    else:

                        df2 = pd.DataFrame({
                            "Variable": df.columns,
                            "Number of Missing values": [df[i].isna().sum() for i in df.columns],
                            "Percentage of Missing values": [df[i].isna().sum() / len(df) * 100 for i in df.columns]
                        })

                        st.table(df2)

                    NaN_replace = st.selectbox("Do you want to replace the missing values ?", ["No", "Yes"])

                    if NaN_replace == "Yes":

                        method_choice = st.selectbox("Do you want to apply the same method to all features ?", ["No", "Yes"])

                        if method_choice == "Yes":

                            Replace_method = st.selectbox(f"Which method do you want to use to replace the missing values ?", ["None", "Mean", "Median", "Mode"])

                            for i in df.columns:

                                    if df[i].isnull().any():

                                        if df[i].dtype != "object":                                    

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

                                            imputer = SimpleImputer(strategy='most_frequent')
                                                    
                                        df[[i]] = imputer.fit_transform(df[[i]])

                            if Replace_method == "None":

                                st.subheader(f"You didn't choose any method to fill the missing values")

                            else:

                                st.subheader(f"You chose the {Replace_method} method to fill the missing values")

                        else:    

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
                
                categorical_features = df.select_dtypes(include=['object']).columns
                label = preprocessing.LabelEncoder()
                for col in categorical_features:
                    df_norm[col] = label.fit_transform(df_norm[col].astype(str))

                    mapping = dict(zip(label.classes_, label.transform(label.classes_)))
                    st.write(f"Mapping for {col}:")
                    mapping_df = pd.DataFrame(list(mapping.items()), columns=['Original', 'Encoded'])
                    st.table(mapping_df.head(5))
                

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
                st.session_state.df_norm = df

            else:

                st.write(df_norm)
                st.session_state.df_norm = df

        except Exception as e:
            st.error(f"Error loading CSV file: Try to change the File Encoding : str{e}")
    else:
        st.write("Please upload a CSV file.")


with tab1 :
            
    if uploaded_file is not None:

        if 'df_norm' in st.session_state:
            df = st.session_state.df_norm

        # ------------------------------------------- VISUALIZATIONS ---------------------------------

        st.header('Data Visualizations')

        if st.checkbox("Create a new feature"):

            operation = st.selectbox("Select Operation", ["average", "groupby_range"])

            if operation == "average":
                column_name = st.selectbox("Select Column for Operation", df.columns)

            if operation == "groupby_range":
                groupby_column = st.selectbox("Select Column for Grouping", df.columns)
                interval = st.number_input("Specify the interval for grouping", min_value=1, value=10)

            new_column_name = st.text_input("New Column Name")
 
            if st.button("Add Column"):

                if operation == "average":
                    average_value = df[column_name].mean()
                    df[new_column_name] = average_value

                elif operation == "groupby_range":
                    df[new_column_name] = pd.cut(df[groupby_column], 
                                         bins=range(0, df[groupby_column].max() + interval, interval), 
                                         right=False, 
                                         labels=[f'{i}-{i+interval-1}' for i in range(0, df[groupby_column].max(), interval)])
                else:
                    st.error("Invalid operation selected")
                
                st.write("Updated DataFrame:")
                st.write(df)
                st.session_state.df = df


        st.subheader('Analysing one feature')


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
            top_values = value_counts.head(10)
            fig_pie = px.pie(top_values, names=selected_column, values='count', title=f'Pie Chart of {selected_column} (top values)')
            st.plotly_chart(fig_pie)

            
            # Multiple columns
            st.subheader('Analysing and comparing multiple features')

            option = st.selectbox("Select an option:", ["Scatter Plot", "Bar Plot", "Violin Plot", "Box Plot"])

            if (len(df.columns)>=2):

                columns = st.multiselect("Select columns for Scatter and Bar Plot", df.columns, max_selections=2)
                if len(columns) >= 2:
                    if option == "Scatter Plot":

                        fig_scatter = px.scatter(df, x=columns[0], y=columns[1],
                                                    title=f'Scatter Plot: {columns[0]} vs {columns[1]}')
                        st.plotly_chart(fig_scatter)

                    elif option == "Bar Plot":
                        fig_bar = px.histogram(df, x=columns[0], y=columns[1],
                                                    title=f'Bar Plot: {columns[0]} vs {columns[1]}')
                        st.plotly_chart(fig_bar)

                    elif option == "Violin Plot":
                        fig_bar = px.violin(df, x=columns[0], y=columns[1],
                                                    title=f'Violin Plot: {columns[0]} vs {columns[1]}')
                        st.plotly_chart(fig_bar)

                    elif option == "Box Plot":
                        fig_bar = px.box(df, x=columns[0], y=columns[1],
                                                    title=f'Box Plot: {columns[0]} vs {columns[1]}')
                        st.plotly_chart(fig_bar)


        if st.checkbox("Heatmap"):
            st.subheader("Heatmap of Correlations")

            corr_matrix = df.corr()
            st.write("Correlation Matrix:")
            st.write(corr_matrix)

            fig_heatmap = px.imshow(corr_matrix, 
                                title='Heatmap of Correlations',
                                labels=dict(x="Features", y="Features"),
                                )
            st.plotly_chart(fig_heatmap)

        st.markdown("---")



with tab2 :
            
    if uploaded_file is not None:

        if not df.isna().any().any():

            if 'df_norm' in st.session_state:
                df = st.session_state.df_norm

            # -------------------------------------------- CLUSTERING OR PREDICTION ---------------------------------
            st.header('Clustering or prediction')

            task = st.selectbox("Select a task", ["Clustering", "Prediction"])
            
            if task == "Clustering":
                st.subheader('Clustering')
                clustering_choice = st.selectbox("Select a clustering algorithm", ["K-Means", "DB-SCAN"])

                if clustering_choice == "K-Means":

                    distortions = []
                    inertias = []
                    mapping1 = {}
                    mapping2 = {}
                    k_test = range(1, 10)

                    for i in k_test:
                        k_test_model = KMeans(n_clusters=i).fit(df)
                        distortions.append(sum(np.min(cdist(df, k_test_model.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])
                        inertias.append(k_test_model.inertia_)

                        mapping1[i] = sum(np.min(cdist(df, k_test_model.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0]
                        mapping2[i] = k_test_model.inertia_

                    fig, ax = plt.subplots()
                    ax.plot(k_test, distortions, 'bx-')
                    ax.set_xlabel('k')
                    ax.set_ylabel('Distortion')
                    ax.set_title('The Elbow Method showing the optimal k')
                    st.pyplot(fig)

                    k = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
                    rs = st.number_input("Random state", value=42)
                    ni = st.selectbox("Number of initializations", ['auto', 1, 5, 10, 20])

                    kmeans = KMeans(n_clusters=k, random_state=rs, n_init=ni)
                    clusters = kmeans.fit(df)
                    labels = clusters.labels_
                    df['Cluster'] = labels
                    inertia = clusters.inertia_
                    centers = clusters.cluster_centers_
                    cluster_counts = df['Cluster'].value_counts().sort_index()
                    cluster_centers = df.groupby('Cluster').mean()

                    # Statistics
                    st.write("Inertia: ", inertia)
                    st.write("Cluster Labels:")
                    st.write(f"{labels}")
                    st.write("Number of data points in each cluster:")
                    st.write(cluster_counts)
                    st.write("Cluster centers (mean of features):")
                    st.write(cluster_centers)

                    # Visualisation

                    if (len(df.columns)>=2):
                        columns = st.multiselect("Select columns for Visualization", df.columns, max_selections=3)
                        if len(columns) == 2:

                            fig = px.scatter(df, x=columns[0], y=columns[1], color='Cluster', title='Cluster Visualization (2D)')
                            st.plotly_chart(fig)

                        elif len(columns) >= 3:
                            fig = px.scatter_3d(df, x=columns[0], y=columns[1], z=columns[2],
                                        color='Cluster', symbol='Cluster',
                                        title=f'3D Scatter Plot of Clusters (K={k})')
                            st.plotly_chart(fig)
                


                elif clustering_choice == "DB-SCAN":

                    nbrs = NearestNeighbors(n_neighbors = 5).fit(df)
                    neigh_dist, neigh_ind = nbrs.kneighbors(df)
                    sort_neigh_dist = np.sort(neigh_dist, axis = 0)
                    k_dist = sort_neigh_dist[:, 4]
                    
                    fig, ax = plt.subplots()
                    ax.plot(k_dist)
                    ax.set_xlabel('Points sorted by distance')
                    ax.set_ylabel('5th Nearest Neighbor Distance')
                    st.pyplot(fig)

                    epsilon = st.number_input("Epsilon", value=0.5)
                    ms = st.number_input("Minimum samples", min_value=1, value=5)

                    dbscan = DBSCAN(eps=epsilon, min_samples=ms)
                    clustering = dbscan.fit(df)
                    labels = clustering.labels_
                    df['Clustering'] = labels
                    st.write(dbscan)
                    st.write(f"{labels}")


                    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise_ = list(labels).count(-1)

                    st.write("Estimated number of clusters: %d" % n_clusters_)
                    st.write("Estimated number of noise points: %d" % n_noise_)


                    if (len(df.columns)>=2):
                        columns = st.multiselect("Select columns for visualization", df.columns, max_selections=3)
                        if len(columns) == 2:

                            fig_2d = px.scatter(df, x=columns[0], y=columns[1], color='Clustering',
                                                    title='Cluster Visualization (2D)')
                            st.plotly_chart(fig_2d)

                        elif len(columns) >= 3:
                            
                            fig_3d = px.scatter_3d(df, x=columns[0], y=columns[1], z=columns[2], color='Clustering',
                                                    title='3D Scatter Plot of Clusters')
                            st.plotly_chart(fig_3d)


            elif task == "Prediction":
                st.subheader('Prediction')

                prediction_choice = st.selectbox("Select a prediction algorithm", ["Regression", "Classification"])

                target_column = st.selectbox("Select target column", df.columns, key="target_column")
                features = df.drop(columns=[target_column])
                target = df[target_column]

                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

                if prediction_choice == "Regression":
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)
                    r2 = model.score(X_test, y_test)

                    st.write(f"Mean Squared Error: {mse}")
                    st.write(f"R^2 Score: {r2}")

                    fig = px.scatter(
                    x=y_test,
                    y=predictions,
                    trendline="ols",
                    labels={'x': 'Real values', 'y': 'Predictions'},
                    title="Prediction vs Real values"
                    )
                    fig.update_traces(marker=dict(size=12, color='rgba(255,182,193, .9)'),
                                    line=dict(width=2, color='rgba(48, 210, 254, 0.8)'))
                    fig.update_layout(showlegend=True)
                    st.plotly_chart(fig)


                elif prediction_choice == "Classification":

                    prediction_method = st.selectbox("Select a prediction method", ["KNN", "Random Forest"], key="prediction_method")

                    if prediction_method == "KNN":

                        nbrs = st.slider("Select the number of neighbors", min_value=1, max_value=100, value=5)
                        classifier = KNeighborsClassifier(n_neighbors=nbrs)
                        classifier.fit(X_train, y_train)


                    elif prediction_method == "Random Forest":
                        
                        max_depth = st.slider("Max depth of the trees", min_value=1, max_value=20, value=5)
                        n_estimators = st.slider("Number of trees", min_value=10, max_value=200, value=100)
                        classifier = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
                        classifier.fit(X_train, y_train)

                    y_pred = classifier.predict(X_test)

                    st.write(pd.crosstab(y_test, y_pred, margins=True))

                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"The accuracy of the model is {accuracy}")

                    results = pd.DataFrame({
                        'Real values': y_test,
                        'Predictions': y_pred
                    })

                    fig = px.scatter(
                        results,
                        x='Real values',
                        y='Predictions',
                        trendline="ols",
                        labels={'x': 'Real values', 'y': 'Predictions'},
                        title="Prediction vs Real values"
                    )
                    fig.update_traces(marker=dict(size=12, color='rgba(255,182,193, .9)'),
                                    line=dict(width=2, color='rgba(48, 210, 254, 0.8)'))
                    fig.update_layout(showlegend=True)
                    st.plotly_chart(fig)