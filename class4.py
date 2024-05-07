import mlflow
logged_model = "runs:/05046420b52842b49c93a606247db7c3/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = pd.read_csv("houses_X.csv", index_col=0)
predicted = loaded_model.predict(data)

data["predicted"] = predicted
data.to_csv("prices.csv")