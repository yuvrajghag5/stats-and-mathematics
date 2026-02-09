import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

data = fetch_california_housing()

x = data.data
y = data.target
print(data.feature_names)

x_train , x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


def eval_model(model, x_train, x_test, y_train, y_test, model_name):
    
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n{model_name}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f} \n")


    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.show()

    return mse, r2, rmse



dt = DecisionTreeRegressor(random_state = 42)
dt_result = eval_model(dt, x_train, x_test, y_train, y_test, "Decision Tree Regressor")

rf = RandomForestRegressor(n_estimators = 100, random_state = 42, n_jobs = 1)
rf_result = eval_model(rf, x_train, x_test, y_train, y_test, "Random Forest Regressor")

et = ExtraTreesRegressor(n_estimators = 100, random_state = 42, n_jobs = 1)
et_result = eval_model(et, x_train, x_test, y_train, y_test, "Extra Trees Regressor")






