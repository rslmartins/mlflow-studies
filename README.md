https://mlflow.org

# Initialize mlflow UI
mlflow ui

# Open https://localhost:5000/

# Run MLFlow through a git project
mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=0.5
mlflow run --no-conda https://github.com/mlflow/mlflow-example.git -P alpha=0.5
# Open https://localhost:5000/
# On this webpage it is possible to transform into either PySpark or Pandas DataFrame for e.g.
# It wil generate the folder mlrun

# Cookiecutter
pip install cookiecutter
cookiecutter https://github.com/jcalvesoliveira/cookiecutter-ds-basic.git

# Class 4
# using a run ID
mlflow models serve -m "runs:/05046420b52842b49c93a606247db7c3/model" -p 5001 --no-conda

# Class 5
mlflow server --backend-store-uri sqlite:///./mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 -p 5000
# Open http://localhost:5000/ and select experiment 'house-prices-script'
# Select the expriment bar and click on Register Model as "Prices"
# Open http://localhost:5000/#/models
# Run python3 class5.py --max-depth 5
# Register the new model under "Prices", it will generate 2 versions
# Set stage of VERSION 2 as Production
# Set environment variable
# export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
# mlflow models serve -m "models:/Prices/Production" -p 5001 --no-conda
# python3 class5.1.py
# mlflow models build-docker -m "models:/Prices/Production" -n "house-prices"