from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np


def eval_trend_model_fit(y_true:pd.DataFrame, y_pred: pd.DataFrame):
    """Custom evaluation function for multistep predictions. Returns average mean absolute error and mean root mean squared error."""
    mae_list, rmse_list = [], []

    for i in range(y_true.shape[0]):
        mae_list.append(
            mean_absolute_error(y_true.iloc[i,:], y_pred.iloc[i,:]))
        rmse_list.append(
            np.sqrt(mean_squared_error(y_true.iloc[i,:], y_pred.iloc[i,:])))
   
    return {
        "mean_mae": np.array(mae_list).mean(), 
        "mean_rmse": np.array(rmse_list).mean()
}   
