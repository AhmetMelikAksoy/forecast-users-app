import datetime as dt
from typing import Dict
from argparse import Namespace
import numpy as np

from fastapi import FastAPI, Request
from http import HTTPStatus
from functools import wraps

from forecast import main, data


app = FastAPI(
    title="Forecast number of users.",
    version="0.1"
)

@app.on_event("startup")
def load_artifacts():
    global artifacts
    global artifacts_total
    artifacts = main.load_artifacts()
    artifacts_total = main.load_artifacts_total_user()


def construct_response(f):

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": dt.datetime.now().isoformat(),
            "url": request.url._url
        }

        if "data" in results:
            response["data"] = results["data"]
        return response
    return wrap


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.get("/performance_next_six_hours", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: str = None) -> Dict:
    """Get the performance metrics."""
    performance = vars(artifacts["performance"])
    data = {"performance": performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response

@app.get("/performance_total_users", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: str = None) -> Dict:
    """Get the performance metrics."""
    performance = vars(artifacts_total["performance"])
    data = {"performance": performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response


@app.get("/args_next_six_hours", tags=["Arguments"])
@construct_response
def _args(request: Request) -> Dict:
    """Get all arguments used for the runs."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "args": vars(artifacts["xgb_args"]),
        },
    }
    return response


@app.get("/args_total_users", tags=["Arguments"])
@construct_response
def _args(request: Request) -> Dict:
    """Get all arguments used for the runs."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "args": vars(artifacts_total["total_user_args"]),
        },
    }
    return response


@app.get("/predict_next_six_hours", tags=["Prediction"])
@construct_response
def _predict_next_six_hours(request: Request) -> Dict:
    """Predict number of users using the latest available data."""

    df = data.elt_data()
    predictions = main.predict_six_hours_ahead(df=df.iloc[-24:,:])
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": predictions
    }
    return response


@app.get("/predict_total_users", tags=["Prediction"])
@construct_response
def _predict_total_users(request: Request) -> Dict:
    """Predict number of users using the latest available data for the next 24 hours."""

    df = data.elt_data()
    predictions = main.predict_total_users(df.iloc[-120:,:])
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": predictions
    }
    return response