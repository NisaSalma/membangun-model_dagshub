import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os


mlflow.set_tracking_uri(
    "https://dagshub.com/NisaSalma/Experimen-SML.mlflow/"
)

mlflow.set_experiment("Ames Housing-Basic Model")

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

mlflow.set_experiment("Ames Housing - Basic Model")

with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2:", r2)