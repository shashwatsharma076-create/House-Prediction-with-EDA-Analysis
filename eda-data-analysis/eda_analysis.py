import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def create_dataset():
    np.random.seed(42)
    n_samples = 1000
    
    categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books']
    regions = ['North', 'South', 'East', 'West']
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.normal(50000, 15000, n_samples).clip(20000, 150000),
        'purchase_amount': np.random.exponential(200, n_samples).clip(10, 2000),
        'product_category': np.random.choice(categories, n_samples),
        'rating': np.random.uniform(1, 5, n_samples).round(1),
        'quantity': np.random.randint(1, 10, n_samples),
        'region': np.random.choice(regions, n_samples),
        'loyalty_years': np.random.randint(0, 15, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    df['total_spent'] = df['purchase_amount'] * df['quantity']
    df['discount'] = np.where(df['loyalty_years'] > 5, df['total_spent'] * 0.15, 
                              np.where(df['loyalty_years'] > 2, df['total_spent'] * 0.08, 0))
    df['final_price'] = df['total_spent'] - df['discount']
    
    missing_idx = np.random.choice(n_samples, 50, replace=False)
    df.loc[missing_idx, 'income'] = np.nan
    
    outlier_idx = np.random.choice(n_samples, 20, replace=False)
    df.loc[outlier_idx, 'purchase_amount'] *= 5
    
    os.makedirs('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/eda-data-analysis/data', exist_ok=True)
    df.to_csv('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/eda-data-analysis/data/sales_data.csv', index=False)
    
    return df

def basic_data_exploration(df):
    print("=" * 60)
    print("BASIC DATA EXPLORATION")
    print("=" * 60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    
    print("\n" + "-" * 40)
    print("Column Information:")
    print("-" * 40)
    print(df.dtypes)
    
    print("\n" + "-" * 40)
    print("First 10 Rows:")
    print("-" * 40)
    print(df.head(10))

def missing_values_analysis(df):
    print("\n" + "=" * 60)
    print("MISSING VALUES ANALYSIS")
    print("=" * 60)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    missing.plot(kind='bar', color='coral')
    plt.title('Missing Values by Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Count')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    ax = df.isnull().sum().plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Missing Values Distribution')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/eda-data-analysis/visualizations/missing_values.png', dpi=150)
    plt.show()

def statistical_summary(df):
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("\nDescriptive Statistics:")
    print(df[numeric_cols].describe().round(2))
    
    print("\n" + "-" * 40)
    print("Categorical Summary:")
    print("-" * 40)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())

def visualizations(df):
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    os.makedirs('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/eda-data-analysis/visualizations', exist_ok=True)
    
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    numeric_cols = ['age', 'income', 'purchase_amount', 'rating', 'quantity', 'total_spent']
    
    for idx, col in enumerate(numeric_cols):
        ax = axes1[idx // 3, idx % 3]
        df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/eda-data-analysis/visualizations/distributions.png', dpi=150)
    plt.show()
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    
    df.boxplot(column='purchase_amount', by='product_category', ax=axes2[0, 0])
    axes2[0, 0].set_title('Purchase Amount by Category')
    axes2[0, 0].set_xlabel('Product Category')
    axes2[0, 0].set_ylabel('Purchase Amount')
    plt.suptitle('')
    
    df.boxplot(column='age', by='region', ax=axes2[0, 1])
    axes2[0, 1].set_title('Age Distribution by Region')
    axes2[0, 1].set_xlabel('Region')
    axes2[0, 1].set_ylabel('Age')
    plt.suptitle('')
    
    df.boxplot(column='income', by='product_category', ax=axes2[1, 0])
    axes2[1, 0].set_title('Income by Product Category')
    axes2[1, 0].set_xlabel('Product Category')
    axes2[1, 0].set_ylabel('Income')
    plt.suptitle('')
    
    axes2[1, 1].boxplot([df['rating']], labels=['Rating'])
    axes2[1, 1].set_title('Rating Distribution')
    axes2[1, 1].set_ylabel('Rating')
    
    plt.tight_layout()
    plt.savefig('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/eda-data-analysis/visualizations/boxplots.png', dpi=150)
    plt.show()
    
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
    
    cat_for_bar = 'product_category'
    cat_counts = df[cat_for_bar].value_counts()
    cat_counts.plot(kind='bar', ax=axes3[0, 0], color='skyblue', edgecolor='black')
    axes3[0, 0].set_title('Purchases by Category')
    axes3[0, 0].set_xlabel('Category')
    axes3[0, 0].set_ylabel('Count')
    axes3[0, 0].tick_params(axis='x', rotation=45)
    
    region_counts = df['region'].value_counts()
    region_counts.plot(kind='pie', ax=axes3[0, 1], autopct='%1.1f%%', startangle=90)
    axes3[0, 1].set_title('Purchases by Region')
    axes3[0, 1].set_ylabel('')
    
    axes3[1, 0].scatter(df['age'], df['purchase_amount'], alpha=0.3)
    axes3[1, 0].set_title('Age vs Purchase Amount')
    axes3[1, 0].set_xlabel('Age')
    axes3[1, 0].set_ylabel('Purchase Amount')
    
    axes3[1, 1].scatter(df['income'], df['total_spent'], alpha=0.3, c=df['rating'], cmap='viridis')
    axes3[1, 1].set_title('Income vs Total Spent')
    axes3[1, 1].set_xlabel('Income')
    axes3[1, 1].set_ylabel('Total Spent')
    plt.colorbar(axes3[1, 1].collections[0], ax=axes3[1, 1], label='Rating')
    
    plt.tight_layout()
    plt.savefig('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/eda-data-analysis/visualizations/scatter_plots.png', dpi=150)
    plt.show()

def correlation_analysis(df):
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/eda-data-analysis/visualizations/correlation_heatmap.png', dpi=150)
    plt.show()

def outlier_detection(df):
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION")
    print("=" * 60)
    
    numeric_cols = ['purchase_amount', 'income', 'age']
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        print(f"\n{col}:")
        print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"  Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
        print(f"  Number of Outliers: {len(outliers)}")

def group_analysis(df):
    print("\n" + "=" * 60)
    print("GROUP ANALYSIS")
    print("=" * 60)
    
    print("\nAverage Purchase by Category:")
    print(df.groupby('product_category')['purchase_amount'].mean().round(2))
    
    print("\nAverage Spending by Region:")
    print(df.groupby('region')['total_spent'].mean().round(2))
    
    print("\nRating by Product Category:")
    print(df.groupby('product_category')['rating'].mean().round(2))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    df.groupby('product_category')['purchase_amount'].mean().plot(kind='bar', ax=axes[0], color='teal')
    axes[0].set_title('Avg Purchase by Category')
    axes[0].tick_params(axis='x', rotation=45)
    
    df.groupby('region')['total_spent'].mean().plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_title('Avg Spending by Region')
    axes[1].tick_params(axis='x', rotation=45)
    
    df.groupby('product_category')['rating'].mean().plot(kind='bar', ax=axes[2], color='purple')
    axes[2].set_title('Avg Rating by Category')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/eda-data-analysis/visualizations/group_analysis.png', dpi=150)
    plt.show()

def main():
    print("Creating synthetic sales dataset...")
    df = create_dataset()
    
    print("\nPerforming basic data exploration...")
    basic_data_exploration(df)
    
    print("\nAnalyzing missing values...")
    missing_values_analysis(df)
    
    print("\nComputing statistical summary...")
    statistical_summary(df)
    
    print("\nCreating visualizations...")
    visualizations(df)
    
    print("\nPerforming correlation analysis...")
    correlation_analysis(df)
    
    print("\nDetecting outliers...")
    outlier_detection(df)
    
    print("\nPerforming group analysis...")
    group_analysis(df)
    
    print("\n" + "=" * 60)
    print("EDA PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nVisualizations saved in: visualizations/")

if __name__ == "__main__":
    main()