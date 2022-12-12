from typing import Dict
from pathlib import Path
from argparse import Namespace
import datetime as dt
import joblib

import pandas as pd
import numpy as np

from forecast import data, train, utils
from config import config


def train_run(
    args_fp: str = "config/args.json" 
) -> None:
    """Run training process for the next_six_hours predictions."""
    df = data.elt_data()
    args = Namespace(**utils.load_dict(filepath=args_fp))
    artifacts_trend_model = train.train_trend_model(df=df, args=args)
    artifacts_complex_model = train.train_complex_model(
        df=df,
        args=args,
        y_fit = artifacts_trend_model["y_fit"],
        y_pred= artifacts_trend_model["y_pred"]
    )

    #save artifacts
    utils.save_dict(artifacts_trend_model["args"], 
                    Path(config.ARTIFACTS_DIR, "trend_args.json"))
    utils.save_dict(artifacts_trend_model["args"], 
                    Path(config.ARTIFACTS_DIR, "xgb_args.json"))
    joblib.dump(artifacts_trend_model["pipeline"], 
                    Path(config.ARTIFACTS_DIR, "trend_model_pipe.pkl"))
    joblib.dump(artifacts_complex_model["pipeline"], 
                    Path(config.ARTIFACTS_DIR, "xgb_model_pipe.pkl"))
    utils.save_dict(artifacts_trend_model["performance"], 
                    Path(config.ARTIFACTS_DIR, "trend_performance.json"))
    utils.save_dict(artifacts_complex_model["performance"], 
                    Path(config.ARTIFACTS_DIR, "xgb_performance.json"))


def load_artifacts() -> Dict:
    """"Load the artifacts for the next_six_hours model."""
    trend_args = Namespace(**utils.load_dict(filepath=Path(config.ARTIFACTS_DIR, "trend_args.json")))
    xgb_args = Namespace(**utils.load_dict(filepath=Path(config.ARTIFACTS_DIR, "xgb_args.json")))
    trend_model = joblib.load(Path(config.ARTIFACTS_DIR, "trend_model_pipe.pkl"))
    xgb_model = joblib.load(Path(config.ARTIFACTS_DIR, "xgb_model_pipe.pkl"))
    xgb_performance = Namespace(**utils.load_dict(filepath=Path(config.ARTIFACTS_DIR, "xgb_performance.json")))

    return {
        "trend_args": trend_args,
        "xgb_args": xgb_args,
        "trend_model_pipe": trend_model,
        "xgb_model_pipe": xgb_model,
        "performance": xgb_performance
    }


def load_artifacts_total_user() -> Dict:
    """"Load the artifacts for the total_users model."""
    total_user_args = Namespace(**utils.load_dict(filepath=Path(config.ARTIFACTS_DIR, "total_user_args.json")))
    model_linear = joblib.load(Path(config.ARTIFACTS_DIR, "total_user_linear_model.pkl"))
    model_tbats = joblib.load(Path(config.ARTIFACTS_DIR, "total_user_tbats.pkl"))
    performance = Namespace(**utils.load_dict(filepath=Path(config.ARTIFACTS_DIR, "total_user_performance.json")))

    return {
        "total_user_args": total_user_args,
        "model_linear": model_linear,
        "model_tbats": model_tbats,
        "performance": performance
    }


def predict_six_hours_ahead(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Prediction process including data preparation. Returns predictions using an ensemble approach."""
    artifacts = load_artifacts()

    trend_inputs = data.create_inputs_for_trend_model(df.index, 
                        args_fp=Path(config.CONFIG_DIR, "args.json"))
    y_trend = artifacts["trend_model_pipe"].predict(trend_inputs)[-1,:]
    
    lft = data.LagFeatureTransformer(artifacts["xgb_args"].target_column)
    X = lft.transform(df, lag_steps=23)
    y_resid = artifacts["xgb_model_pipe"].predict(X)


    return {
        "predictions": pd.DataFrame((y_trend + y_resid).reshape(-1,1))
    }


def predict_total_users(
    y: pd.DataFrame):
    """Prediction process for the total number of users model."""
    #load artifacts
    artifacts = load_artifacts_total_user()
    #arrange the index to include 24 hours ahead
    multi_index = y.index.append(
                            y[-24:].index+dt.timedelta(days=1))

    trend_inputs = data.create_inputs_for_trend_model(
        index=multi_index,
        args_fp=Path(config.CONFIG_DIR, "args_total_user.json")
    )

    y_trend = artifacts["model_linear"].predict(trend_inputs)[-24:]
    y_tbats = artifacts["model_tbats"].forecast(steps=24)

    y_preds = (y_trend + y_tbats)/2
    y_preds = y_preds.astype(int)
    total = np.sum(y_preds)
    return {
        "predictions": int(total)
    }

    
if __name__ == "__main__":

    from utils import track
    @track
    def run_all():
        df = data.elt_data()
        train_run()
        preds = predict_six_hours_ahead(df.iloc[-24:,:])
        print("Predictions for the next six hours:")
        print(preds)

        train.run_train_total_user_model()
        predict_total = predict_total_users(df.iloc[-120:,0])
        print(f"Total users for the next 24 hours: {np.sum(predict_total['predictions'])}")

    run_all()