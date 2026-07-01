# ==========================================================
# SALES PREDICTION USING REGRESSION
# UoPeople - Data Mining and Machine Learning - Week 2
# Mohpheth Ekhaguere
# ==========================================================

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ==========================================================
# Create Dataset
# ==========================================================

data = {
    "Advertising_Spend": [2000, 3000, 2500, 4000, 3500, None, 5000, 4500, 3000, 3800],
    "Store_Size": [1500, 2000, 1800, 2200, None, 2100, 2500, 2400, 2000, 2300],
    "Customers": [200, 250, 230, 300, 280, 260, 320, 310, 270, None],
    "Promotion": ["Yes", "No", "Yes", "Yes", "No", "No", "Yes", "Yes", "No", "Yes"],
    "Sales": [40000, 50000, 45000, 60000, 52000, 48000, 65000, 63000, 51000, 59000]
}

df = pd.DataFrame(data)

print("\n==============================")
print("ORIGINAL DATASET")
print("==============================\n")

print(df)

# ==========================================================
# Missing Values
# ==========================================================

print("\n==============================")
print("MISSING VALUES")
print("==============================\n")

print(df.isnull().sum())

# ==========================================================
# Data Types
# ==========================================================

print("\n==============================")
print("DATA TYPES")
print("==============================\n")

print(df.dtypes)

# ==========================================================
# Dataset Information
# ==========================================================

print("\n==============================")
print("DATASET INFORMATION")
print("==============================\n")

print(df.info())

# ==========================================================
# Summary Statistics
# ==========================================================

print("\n==============================")
print("SUMMARY STATISTICS")
print("==============================\n")

print(df.describe())

# ==========================================================
# DATA PREPROCESSING
# ==========================================================

# Fill missing numerical values using the column mean.
# Mean is appropriate because only a few values are missing.

df["Advertising_Spend"] = df["Advertising_Spend"].fillna(
    df["Advertising_Spend"].mean()
)

df["Store_Size"] = df["Store_Size"].fillna(
    df["Store_Size"].mean()
)

df["Customers"] = df["Customers"].fillna(
    df["Customers"].mean()
)

# Convert Promotion column to numerical values

df["Promotion"] = df["Promotion"].map({
    "Yes": 1,
    "No": 0
})

print("\n==============================")
print("DATASET AFTER CLEANING")
print("==============================\n")

print(df)

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

numerical_columns = [
    "Advertising_Spend",
    "Store_Size",
    "Customers"
]

df[numerical_columns] = scaler.fit_transform(
    df[numerical_columns]
)

print("\n==============================")
print("SCALED DATASET")
print("==============================\n")

print(df)

# ==========================================================
# Define Features and Target
# ==========================================================

X = df.drop("Sales", axis=1)

y = df["Sales"]

# ==========================================================
# Train/Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n==============================")
print("TRAINING SET SIZE:", len(X_train))
print("TEST SET SIZE:", len(X_test))
print("==============================")

# ==========================================================
# SIMPLE LINEAR REGRESSION
# ==========================================================

X_simple_train = X_train[["Advertising_Spend"]]

X_simple_test = X_test[["Advertising_Spend"]]

simple_model = LinearRegression()

simple_model.fit(
    X_simple_train,
    y_train
)

simple_predictions = simple_model.predict(
    X_simple_test
)

print("\n==============================")
print("SIMPLE LINEAR REGRESSION")
print("==============================")

print("Coefficient:")
print(simple_model.coef_)

print("Intercept:")
print(simple_model.intercept_)

# ==========================================================
# MULTIPLE LINEAR REGRESSION
# ==========================================================

multiple_model = LinearRegression()

multiple_model.fit(
    X_train,
    y_train
)

multiple_predictions = multiple_model.predict(
    X_test
)

print("\n==============================")
print("MULTIPLE LINEAR REGRESSION")
print("==============================")

print("Coefficients:")
print(multiple_model.coef_)

print("Intercept:")
print(multiple_model.intercept_)

# ==========================================================
# POLYNOMIAL REGRESSION
# ==========================================================

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly_train = poly.fit_transform(X_train)

X_poly_test = poly.transform(X_test)

poly_model = LinearRegression()

poly_model.fit(
    X_poly_train,
    y_train
)

poly_predictions = poly_model.predict(
    X_poly_test
)

# ==========================================================
# RIDGE REGRESSION
# ==========================================================

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(
    X_train,
    y_train
)

ridge_predictions = ridge_model.predict(
    X_test
)

print("\n==============================")
print("RIDGE REGRESSION")
print("==============================")

print("Coefficients:")
print(ridge_model.coef_)

print("Intercept:")
print(ridge_model.intercept_)

# ==========================================================
# LASSO REGRESSION
# ==========================================================

lasso_model = Lasso(alpha=1.0)

lasso_model.fit(
    X_train,
    y_train
)

lasso_predictions = lasso_model.predict(
    X_test
)

print("\n==============================")
print("LASSO REGRESSION")
print("==============================")

print("Coefficients:")
print(lasso_model.coef_)

print("Intercept:")
print(lasso_model.intercept_)

# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_model(model_name, actual, predicted):

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    print("\n===================================")
    print(model_name)
    print("===================================")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"R2   : {r2:.4f}")

    return rmse, mae, r2


simple_rmse, simple_mae, simple_r2 = evaluate_model(
    "Simple Linear Regression",
    y_test,
    simple_predictions
)

multiple_rmse, multiple_mae, multiple_r2 = evaluate_model(
    "Multiple Linear Regression",
    y_test,
    multiple_predictions
)

poly_rmse, poly_mae, poly_r2 = evaluate_model(
    "Polynomial Regression",
    y_test,
    poly_predictions
)

ridge_rmse, ridge_mae, ridge_r2 = evaluate_model(
    "Ridge Regression",
    y_test,
    ridge_predictions
)

lasso_rmse, lasso_mae, lasso_r2 = evaluate_model(
    "Lasso Regression",
    y_test,
    lasso_predictions
)

# ==========================================================
# NEW SALES PREDICTION
# ==========================================================

print("\n===================================")
print("NEW SALES PREDICTION")
print("===================================")

new_data = pd.DataFrame({

    "Advertising_Spend":[4200],
    "Store_Size":[2100],
    "Customers":[290],
    "Promotion":[1]

})

new_data[numerical_columns] = scaler.transform(
    new_data[numerical_columns]
)

simple_prediction = simple_model.predict(
    new_data[["Advertising_Spend"]]
)

multiple_prediction = multiple_model.predict(
    new_data
)

new_poly = poly.transform(new_data)

poly_prediction = poly_model.predict(
    new_poly
)

ridge_prediction = ridge_model.predict(
    new_data
)

lasso_prediction = lasso_model.predict(
    new_data
)

print(f"Simple Linear Regression : {simple_prediction[0]:,.2f}")

print(f"Multiple Linear Regression : {multiple_prediction[0]:,.2f}")

print(f"Polynomial Regression : {poly_prediction[0]:,.2f}")

print(f"Ridge Regression : {ridge_prediction[0]:,.2f}")

print(f"Lasso Regression : {lasso_prediction[0]:,.2f}")

# ==========================================================
# ACTUAL VS PREDICTED PLOTS
# ==========================================================

plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    simple_predictions
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Simple Linear Regression")

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.show()


plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    multiple_predictions
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Multiple Linear Regression")

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.show()


plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    poly_predictions
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Polynomial Regression")

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.show()


plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    ridge_predictions
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Ridge Regression")

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.show()


plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    lasso_predictions
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Lasso Regression")

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.show()

# ==========================================================
# MODEL COMPARISON
# ==========================================================

results = pd.DataFrame({

    "Model":[
        "Simple Linear",
        "Multiple Linear",
        "Polynomial",
        "Ridge",
        "Lasso"
    ],

    "RMSE":[
        simple_rmse,
        multiple_rmse,
        poly_rmse,
        ridge_rmse,
        lasso_rmse
    ],

    "MAE":[
        simple_mae,
        multiple_mae,
        poly_mae,
        ridge_mae,
        lasso_mae
    ],

    "R2":[
        simple_r2,
        multiple_r2,
        poly_r2,
        ridge_r2,
        lasso_r2
    ]

})

print("\n===================================")
print("MODEL COMPARISON")
print("===================================\n")

print(results)

best_model = results.loc[
    results["R2"].idxmax()
]

print("\n===================================")
print("BEST MODEL")
print("===================================\n")

print(best_model)

print("\n===================================")
print("OVERFITTING COMMENT")
print("===================================\n")

if poly_r2 > multiple_r2 + 0.10:
    print("Polynomial regression may be overfitting because it performs significantly better than the linear models.")
else:
    print("There is no obvious evidence of severe overfitting based on the available evaluation metrics.")

print("\n===================================")
print("PROGRAM COMPLETED SUCCESSFULLY")
print("===================================\n")
