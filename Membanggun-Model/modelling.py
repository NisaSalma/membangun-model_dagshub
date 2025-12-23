import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ===============================
# MLFLOW CONFIG
# ===============================
mlflow.set_tracking_uri(
    "https://dagshub.com/NisaSalma/Experimen-SML.mlflow"
)

mlflow.set_experiment("Ames Housing - Basic Model")


# ===============================
# LOAD DATA
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "ames_preprocessing",
    "ames_preprocessed.csv"
)

print("Dataset path:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# TRAINING WITH AUTOLOG
# ===============================
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForestRegressor"):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2:", r2)