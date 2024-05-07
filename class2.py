import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost
import mlflow

df = pd.read_csv("houses.csv")
X = df.drop("price", axis=1)
y = df["price"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dtrain = xgboost.DMatrix(X_train, label=y_train)
dtest = xgboost.DMatrix(X_test, label=y_test)

xgb_params = {"learning_rate": 0.2, "n_estimators": 50, "random_state": 42}


mlflow.set_experiment("house-prices-script")
with mlflow.start_run():
	mlflow.xgboost.autolog()
	xgb = xgboost.train(xgb_params, dtrain, evals=[(dtrain, "train")])
	xgb_predicted = xgb.predict(dtest)
	mse = mean_squared_error(y_test, xgb_predicted)
	rmse = mse ** 0.5
	r2 = r2_score(y_test, xgb_predicted)
	mlflow.log_metric("mse", mse)
	mlflow.log_metric("rmse", rmse)
	mlflow.log_metric("r2", r2)
