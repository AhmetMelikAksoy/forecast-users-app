from pathlib import Path
from argparse import Namespace
from typing import Dict
import joblib
import datetime as dt

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tbats import TBATS
import xgboost as xgb

from forecast import data, eval, predict, utils
from config import config


def train_trend_model(
    args: Namespace,
    df: pd.DataFrame) -> Dict:
    """Training process for the trend predictions for the next_six_hours model."""
    #for reproducibility
    utils.set_seeds()

    #generate target data with feature steps
    lft = data.LagFeatureTransformer(args.target_column)
    y = lft.transform(df, feature_steps=6).drop(args.target_column, axis=1)
    y = y.drop(args.feature_columns, axis=1)
    
    #generate features for trend model
    X = data.create_inputs_for_trend_model(y.index,
                                        args_fp=Path(config.CONFIG_DIR, "args.json"))

    X_tr, y_tr, X_tst, y_tst = data.get_data_splits(X, y, train_size=args.train_size)

    trend_pipe = Pipeline([
        ('LinearRegression', LinearRegression(fit_intercept=args.fit_intercept))
    ]).fit(X_tr, y_tr)

    y_pred = predict.get_preds(
        X = X_tst,
        pipeline= trend_pipe,
        index=y_tst.index,
        columns=y_tst.columns
    )

    y_fit = predict.get_preds(
        X = X_tr,
        pipeline= trend_pipe,
        index=y_tr.index,
        columns=y_tr.columns
    )

    performance = eval.eval_trend_model_fit(y_tst, y_pred)

    return {
        "args": vars(args),
        "lag_feature_transformer": lft,
        "pipeline": trend_pipe,
        "performance": performance,
        "tr_end_index": y_tr.index[-1],
        "y_pred": y_pred,
        "y_fit": y_fit
    }


def train_complex_model(
    df: pd.DataFrame,
    args: Namespace,
    y_fit: pd.DataFrame,
    y_pred: pd.DataFrame) -> Dict:
    """Training process for the residual predictions for the next_six_hours model."""

    utils.set_seeds()

    #generate target
    lft = data.LagFeatureTransformer(args.target_column)
    y = lft.transform(df, feature_steps=6).drop(args.target_column, axis=1)
    y = y.drop(args.feature_columns, axis=1)

    X = lft.transform(df, lag_steps=23)

    #train-test split
    #we need to be careful in arranging train and test datasets, since dropping null values is different for two models
    tr_end_index = y_fit.index[-1]
    tr_first_index = X.index[0]
    tst_end_index = y.index[-1]

    X_tr = X.loc[:tr_end_index, :]
    X_tst = X.loc[tr_end_index+dt.timedelta(hours=1):tst_end_index,:]
    y_tr = y.loc[tr_first_index:tr_end_index, :] - y_fit.loc[tr_first_index:]
    y_tst = y.loc[tr_end_index+dt.timedelta(hours=1):,:] - y_pred

    xgb_pipeline = Pipeline(
        [
            (
                "binary_encoder", data.BinaryEncoder(args.to_binary)
            ),
            (
                "model", MultiOutputRegressor(xgb.XGBRegressor(
                    **args.xgboost_best_params
                ))
            )
        ]
    ).fit(X_tr, y_tr)

    preds = predict.get_preds(
        X_tst,
        xgb_pipeline,
        index=y_tst.index,
        columns=y_tst.columns
    )

    performance = eval.eval_trend_model_fit(y_tst, preds)

    return {
        "args": vars(args),
        "lag_feature_transformer": lft,
        "pipeline": xgb_pipeline,
        "performance": performance
    }


def run_train_total_user_model(
    args_fp: str = "args_total_user.json"
) -> None:
    """Train process and saving artifacts combined under this function, also it includes two models."""

    utils.set_seeds()

    df = data.elt_data()
    args = Namespace(**utils.load_dict(filepath=Path(config.CONFIG_DIR, args_fp)))

    # multi_index = df.index.append(df.iloc[-24:,:].index+dt.timedelta(days=1))
    X = data.create_inputs_for_trend_model(
        index=df.index,
        args_fp=args_fp
    )
    y = df.iloc[:,0]

    X_tr, X_tst = X.iloc[:-args.test_size,:], X.iloc[-args.test_size:,:]
    y_tr, y_tst = y.iloc[:-args.test_size], y.iloc[-args.test_size:]

    model_linear = LinearRegression(fit_intercept=False)
    model_linear.fit(X_tr, y_tr)

    model_tbats = TBATS(
        seasonal_periods = [24]
    ).fit(y_tr)
    
    preds_linear = model_linear.predict(X_tst)
    preds_tbats = model_tbats.forecast(steps=args.test_size)
    preds = (preds_linear + preds_tbats)/2

    performance = {
        "mae": mean_absolute_error(y_tst, preds),
        "rmse": np.sqrt(mean_squared_error(y_tst, preds))
    }

    utils.save_dict(vars(args), 
                    Path(config.ARTIFACTS_DIR, "total_user_args.json"))
    joblib.dump(model_linear, 
                Path(config.ARTIFACTS_DIR, "total_user_linear_model.pkl"))
    joblib.dump(model_tbats,
                Path(config.ARTIFACTS_DIR, "total_user_tbats.pkl"))
    utils.save_dict(performance, 
                    Path(config.ARTIFACTS_DIR, "total_user_performance.json"))