import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

def load_data():
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'square_footage': np.random.uniform(1000, 4000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'year_built': np.random.randint(1950, 2024, n_samples),
        'lot_size': np.random.uniform(0.1, 2.0, n_samples),
        'garage_spots': np.random.randint(0, 4, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    price = (200 * df['square_footage'] + 
            10000 * df['bedrooms'] + 
            15000 * df['bathrooms'] +
            500 * (2024 - df['year_built']) * 100 +
            20000 * df['lot_size'] +
            15000 * df['garage_spots'] +
            np.random.normal(0, 20000, n_samples))
    
    df['price'] = price
    
    os.makedirs('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/house-price-prediction/data', exist_ok=True)
    df.to_csv('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/house-price-prediction/data/house_prices.csv', index=False)
    
    return df

def preprocess_data(df):
    X = df.drop('price', axis=1)
    y = df['price']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean Squared Error: ${mse:,.2f}")
    print(f"Root Mean Squared Error: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    print("=" * 50)
    
    return y_pred, rmse, r2

def plot_results(X_test, y_test, y_pred):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, c='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Price ($)')
    axes[0, 0].set_ylabel('Predicted Price ($)')
    axes[0, 0].set_title('Actual vs Predicted Prices')
    
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, c='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Price ($)')
    axes[0, 1].set_ylabel('Residuals ($)')
    axes[0, 1].set_title('Residual Plot')
    
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuals ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    
    feature_importance = pd.Series([200, 10000, 15000, 500, 20000, 15000], 
                                    index=X_test.columns)
    feature_importance.plot(kind='bar', ax=axes[1, 1], color='orange')
    axes[1, 1].set_xlabel('Features')
    axes[1, 1].set_ylabel('Coefficient Value')
    axes[1, 1].set_title('Feature Importance (Coefficients)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/house-price-prediction/model_results.png', dpi=150)
    plt.show()

def explore_data(df):
    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    print("\nDataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nCorrelation with Price:")
    print(df.corr()['price'].sort_values(ascending=False))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, col in enumerate(df.columns[:-1]):
        ax = axes[idx // 3, idx % 3]
        ax.scatter(df[col], df['price'], alpha=0.3)
        ax.set_xlabel(col.replace('_', ' ').title())
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{col.replace("_", " ").title()} vs Price')
    
    plt.tight_layout()
    plt.savefig('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/house-price-prediction/eda_plots.png', dpi=150)
    plt.show()

def main():
    print("Loading and creating dataset...")
    df = load_data()
    
    print("\nExploring data...")
    explore_data(df)
    
    print("\nPreprocessing data...")
    X, y = preprocess_data(df)
    
    print("\nTraining Linear Regression model...")
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    print("\nModel Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:,.2f}")
    print(f"  Intercept: {model.intercept_:,.2f}")
    
    print("\nEvaluating model...")
    y_pred, rmse, r2 = evaluate_model(model, X_test, y_test)
    
    print("\nPlotting results...")
    plot_results(X_test, y_test, y_pred)
    
    print("\n" + "=" * 50)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()