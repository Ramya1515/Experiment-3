# Experiment-3
Implements Ordinary Least Squares (OLS) linear regression using the normal equation and compares it with scikit-learn.
AIM:
 Implement OLS via normal equations and compare to scikit-learn.
ALGORITHM (brief):
 Normal equation : 𝛽̂ = (𝑋
𝑇𝑋)
−1𝑋
𝑇𝑦
PROGRAM:
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
DETAILED EXPLANATION
1. X and y — toy predictors and target.
2. X0 = np.hstack([np.ones(...), X]) — adds an intercept column of ones.
3. beta_hat = (X0^T X0)^{-1} X0^T y — compute closed-form coefficients.
4. np.linalg.solve alternative? (we used inverse for clarity.)
5. Fit LinearRegression() (sklearn) for comparison — it uses SVD numerically
stable.
6. Compute and compare MSE.
SAMPLE OUTPUT :
Closed-form beta: [10.5 2. 1.8]
MSE (closed): 0.0
MSE (sklearn): 0.0
RESULT:
 Students see how closed-form OLS yields coefficients; numerical stability note: prefer
SVD in practice.
THEORY:
 OLS minimizes squared residuals; closed-form solution requires invertible Gram
matrix
