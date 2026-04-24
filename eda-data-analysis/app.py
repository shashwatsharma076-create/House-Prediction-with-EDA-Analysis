import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from eda_analysis import create_dataset, basic_data_exploration, missing_values_analysis, statistical_summary

st.set_page_config(page_title="EDA Dashboard", page_icon="📊")

st.title("📊 Exploratory Data Analysis Dashboard")
st.write("Interactive EDA on sales data")

df = create_dataset()

tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "📈 Visualizations", "🔍 Analysis", "📉 Statistics"])

with tab1:
    st.subheader("Raw Data")
    st.dataframe(df.head(10))
    
    st.subheader("Column Info")
    st.write(df.dtypes)
    
    st.subheader("Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

with tab2:
    st.subheader("Distributions")
    
    col = st.selectbox("Select Column", ['age', 'income', 'purchase_amount', 'rating', 'total_spent'])
    fig, ax = plt.subplots()
    df[col].hist(bins=30, ax=ax, edgecolor='black', color='teal')
    st.pyplot(fig)
    
    st.subheader("Box Plot")
    col_box = st.selectbox("Select Column for Box Plot", ['purchase_amount', 'income', 'age', 'total_spent'])
    fig2, ax2 = plt.subplots()
    df.boxplot(column=col_box, ax=ax2)
    st.pyplot(fig2)
    
    st.subheader("Category Distribution")
    fig3, ax3 = plt.subplots()
    df['product_category'].value_counts().plot(kind='bar', ax=ax3, color='coral')
    st.pyplot(fig3)

with tab3:
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.bar_chart(missing)
    
    st.subheader("Correlation with Price")
    corr = df.corr()['total_spent'].sort_values(ascending=True)
    st.bar_chart(corr)
    
    st.subheader("Group Analysis")
    group_by = st.selectbox("Group by", ['product_category', 'region'])
    grouped = df.groupby(group_by)['total_spent'].mean()
    fig, ax = plt.subplots()
    grouped.plot(kind='bar', ax=ax, color='purple')
    st.pyplot(fig)

with tab4:
    st.subheader("Statistical Summary")
    st.write(df.describe())
    
    st.subheader("Categorical Counts")
    cat_col = st.selectbox("Select Category", ['product_category', 'region'])
    st.write(df[cat_col].value_counts())