from __future__ import annotations

from typing import Callable, Iterable, List, Tuple, TypeVar, TypedDict, Union

from matplotlib import axes, pyplot
import numpy
import pandas
from sklearn.metrics import (
    mean_absolute_percentage_error,
    root_mean_squared_error,
    r2_score,
)


__all__ = ()
T = TypeVar("T")
MatrixLike = Union[numpy.ndarray, pandas.DataFrame]


class Metrics(TypedDict):
    rmse: float
    mape: float
    r2: float


def metrics(true: MatrixLike, pred: MatrixLike) -> Metrics:
    return {
        "rmse": root_mean_squared_error(true, pred),
        "mape": mean_absolute_percentage_error(true, pred),
        "r2": r2_score(true, pred),
    }


def plot_context_size_metrics(values: List[float], *, label: str) -> None:
    index = range(1, 1 + len(values))

    axes.Axes.set_xticks(pyplot.gca(), ticks=index)
    pyplot.plot(index, values, label=label)

    pyplot.xlabel("Context size")
    pyplot.ylabel(label)

    pyplot.legend()
    pyplot.show()


def extract_xy(table: pandas.DataFrame, *, context_size: int) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    series: List[pandas.Series] = []
    original = table.columns
    output_names: List[str] = []

    for column in table.columns:
        for shift in range(1, context_size + 1):
            s = table[column].shift(shift)
            s.name = f"{column}-{shift}"

            series.append(s)
            output_names.append(s.name)

    table = pandas.concat([table, *series], axis=1)
    table.dropna(inplace=True)

    return table[output_names], table[original]


def rank_metrics(keys: Iterable[T], *metrics: Callable[[T], float]) -> T:
    keys = list(keys)
    scores = {k: 0 for k in keys}

    for metric in metrics:
        ordered = sorted((metric(k), k) for k in keys)

        for index, (_, k) in enumerate(ordered):
            scores[k] += index

    return min(keys, key=scores.__getitem__)
