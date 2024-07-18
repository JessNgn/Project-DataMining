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

    # ------------------------------------------- PART 3 VISUALIZATIONS ---------------------------------

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

    # ------------------------------------------- PART 4 CLUSTERING OR PREDICTION ---------------------------------
    st.subheader('Clustering or prediction')

    task = st.selectbox("Select a task", ["Clustering", "Prediction"])
    
    if task == "Clustering":
        st.subheader('Clustering')
        clustering_choice = st.selectbox("Select a clustering algorithm", ["K-Means", "DB-SCAN"])

        if clustering_choice == "K-Means":

            from sklearn.cluster import KMeans

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
                columns = st.multiselect("Select columns for Scatter Plot", df.columns, max_selections=3)
                if len(columns) == 2:

                    fig = px.scatter(df, x=columns[0], y=columns[1], color='Cluster', title='Cluster Visualization (2D)')
                    st.plotly_chart(fig)

                elif len(columns) >= 3:
                    fig = px.scatter_3d(df, x=columns[0], y=columns[1], z=columns[2],
                                color='Cluster', symbol='Cluster',
                                title=f'3D Scatter Plot of Clusters (K={k})')
                    st.plotly_chart(fig)
        


        elif clustering_choice == "DB-SCAN":
            from sklearn.cluster import DBSCAN
            epsilon = st.number_input("Epsilon", value=0.5)
            ms = st.number_input("Minimum samples", min_value=1, value=5)

            dbscan = DBSCAN(eps=epsilon, min_samples=ms)
            clustering = dbscan.fit(df)
            labels = clustering.labels_
            df['Clustering'] = labels
            st.write(dbscan)
            st.write(labels)


            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            st.write("Estimated number of clusters: %d" % n_clusters_)
            st.write("Estimated number of noise points: %d" % n_noise_)


            if (len(df.columns)>=2):
                columns = st.multiselect("Select columns for Scatter Plot", df.columns, max_selections=3)
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

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, accuracy_score



        prediction_choice = st.selectbox("Select a prediction algorithm", ["Regression", "Classification"])

        target_column = st.selectbox("Select target column", df.columns)
        features = df.drop(columns=[target_column])
        target = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        if prediction_choice == "Regression":
            from sklearn.linear_model import LinearRegression

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

            




