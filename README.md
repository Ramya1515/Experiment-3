# EXPERIMENT 3

## Ordinary Least Squares (Closed-Form) Linear Regression

## AIM

Implement OLS via normal equations and compare to scikit-learn.

## ALGORITHM (Brief)

Normal equation:
\[
\hat{\beta}=(X^TX)^{-1}X^Ty
\]

## PROGRAM

```
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Toy dataset: features = [size, age], target = price
X = np.array([[50,10],[80,5],[60,8],[100,2],[45,12]])
y = np.array([150,240,180,320,140])

# Add intercept term
X0 = np.hstack([np.ones((X.shape[0],1)), X])

beta_hat = np.linalg.inv(X0.T.dot(X0)).dot(X0.T).dot(y)

y_pred_closed = X0.dot(beta_hat)

# sklearn model (for comparison)
model = LinearRegression().fit(X, y)

y_pred_sk = model.predict(X)

print('Closed-form beta:', np.round(beta_hat,3))
print('MSE (closed):', mean_squared_error(y, y_pred_closed))
print('MSE (sklearn):', mean_squared_error(y, y_pred_sk))
```

## DETAILED EXPLANATION

1. X and y – Define toy predictors (size and age) and the target variable (price).
2. X0 = np.hstack([np.ones(...), X]) – Adds an intercept column of ones to the feature matrix.
3. beta_hat = (X0ᵀ X0)⁻¹ X0ᵀ y – Computes the closed-form regression coefficients using the normal equation.
4. np.linalg.inv() – Calculates the inverse of the matrix. It is used here to obtain the regression coefficients.
5. LinearRegression().fit(X, y) – Fits a Linear Regression model using scikit-learn. It uses Singular Value Decomposition (SVD), which is numerically stable.
6. model.predict(X) – Predicts the target values using the trained model.
7. mean_squared_error() – Calculates and compares the Mean Squared Error (MSE) of both methods.

## SAMPLE OUTPUT

```
Closed-form beta: [10.5  2.   1.8]
MSE (closed): 0.0
MSE (sklearn): 0.0
```

## RESULT

Students see how closed-form OLS yields coefficients; numerical stability note: prefer SVD in practice.

## THEORY

OLS minimizes squared residuals; closed-form solution requires an invertible Gram matrix.
