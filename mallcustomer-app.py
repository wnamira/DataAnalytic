#Import Packages

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
import plotly.tools
file = "mall_customer.csv"
df = pd.read_csv(file)


st.header("My Mall Customer App")

option = st.sidebar.selectbox(
    'Select a mini project',
     ['whole data', 'description','KMeans cluster'])

if option=='whole data':
    whole = df
    st.table(whole)
    
if option=='description':
    chart_data = df.describe()
    st.table(chart_data)

else:
    #Preprocessing
    features = ['Annual_Income_(k$)', 'Spending_Score']
    X = df[features]
    plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score']);
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5);
    st.pyplot()
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
