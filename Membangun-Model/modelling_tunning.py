import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ================= DAGSHUB =================
mlflow.set_tracking_uri(
    "https://dagshub.com/NisaSalma/Membangun_Model.mlflow"
)
mlflow.set_experiment("Ames Housing - Tuning Model")

# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ames_preprocessing", "ames_preprocessed.csv")

df = pd.read_csv(DATA_PATH)

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20]
}

with mlflow.start_run():
    model = RandomForestRegressor(random_state=42)

    grid = GridSearchCV(
        model,
        params,
        cv=3,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # MANUAL LOGGING (OK untuk tuning)
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(best_model, "model")

    print("Best Params:", grid.best_params_)
    print("MSE:", mse)
    print("R2:", r2)