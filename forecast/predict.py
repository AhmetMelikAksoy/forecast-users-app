import pandas as pd
from sklearn.pipeline import Pipeline


def get_preds(
    X: pd.DataFrame,
    pipeline: Pipeline,
    columns=None,
    index=None
) -> pd.DataFrame:
    """Return predictions with pandas dataframe format."""

    trend_preds = pd.DataFrame(
            pipeline.predict(X),
            index=index,
            columns=columns)

    return trend_preds