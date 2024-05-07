import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRFRegressor
import mlflow

df = pd.read_csv("houses.csv")
X = df.drop("price", axis=1)
y = df["price"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mlflow.set_experiment("house-prices-eda")
mlflow.start_run()

lr = LinearRegression()
lr.fit(X_train, y_train)
lr.predict(X_test)[0:2]

mlflow.sklearn.log_model(lr, "Linear Regression")

mlflow.log_metric("MSE for Linear Regression", mean_squared_error(y_test, lr.predict(X_test)))
mlflow.log_metric("RMSE for Linear Regression", mean_squared_error(y_test, lr.predict(X_test)) ** 0.5 )
mlflow.log_metric("R2 for Linear Regression", r2_score(y_test, lr.predict(X_test)))

xgb_params = {"learning_rate": 0.2, "n_estimators": 50, "random_state": 42}
xgb = XGBRFRegressor(**xgb_params)
xgb.fit(X_train, y_train)

mlflow.sklearn.log_model(xgb, "XGBRFRegressor")

mse = mean_squared_error(y_test, xgb.predict(X_test))
rmse = mse ** 0.5
r2 = r2_score(y_test, xgb.predict(X_test))

mlflow.log_metric("MSE for XGBRFRegressor", mse)
mlflow.log_metric("RMSE for XGBRFRegressor", rmse)
mlflow.log_metric("R2 for XGBRFRegressor", r2)

# mlflow.get_experiment_by_name()
# mlflow.list_run_infos('1')
# mlflow.get_run()
