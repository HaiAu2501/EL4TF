from __future__ import annotations

import pandas
import random
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Protocol, Tuple, TypedDict, TYPE_CHECKING

import numpy
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import Metrics, extract_xy, metrics, rank_metrics


def sigmoid(x: numpy.ndarray) -> numpy.ndarray:
    return 1 / (1 + numpy.exp(-x))


class Model(Protocol):
    def predict(self, x: numpy.ndarray) -> numpy.ndarray: ...
    def fit(self, x: numpy.ndarray, y: numpy.ndarray) -> None: ...


class DRVFL(Model):

    __slots__ = ("params", "ridge")
    if TYPE_CHECKING:
        params: List[Tuple[numpy.ndarray, numpy.ndarray]]
        ridge: RidgeCV

    def __init__(self, params: List[Tuple[numpy.ndarray, numpy.ndarray]]) -> None:
        self.params = params
        self.ridge = RidgeCV()

    def _transform_input(self, x: numpy.ndarray) -> numpy.ndarray:
        h = x
        concat = [x]
        for w, b in self.params:
            h = sigmoid(h @ w + b)
            concat.append(h)

        return numpy.concat(concat, axis=1)

    def predict(self, x: numpy.ndarray) -> numpy.ndarray:
        z = self._transform_input(x)
        return self.ridge.predict(z)

    def fit(self, x: numpy.ndarray, y: numpy.ndarray) -> None:
        z = self._transform_input(x)
        self.ridge.fit(z, y)

    @classmethod
    def initialize(
        cls,
        input_size: int,
        *,
        seed: Optional[int] = None,
        hidden_sizes: Iterable[int],
    ) -> DRVFL:
        rng = numpy.random.default_rng(seed)

        results: List[Tuple[numpy.ndarray, numpy.ndarray]] = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            w = rng.random((last_size, hidden_size))
            b = rng.random((1, hidden_size))
            last_size = hidden_size

            results.append((w, b))

        return cls(results)


class EDRVFL(Model):

    __slots__ = ("models",)
    if TYPE_CHECKING:
        models: List[DRVFL]

    def __init__(self, models: List[DRVFL]) -> None:
        self.models = models

    def predict(self, x: numpy.ndarray) -> numpy.ndarray:
        s = [model.predict(x) for model in self.models]
        return numpy.divide(sum(s), len(s))

    def fit(self, x: numpy.ndarray, y: numpy.ndarray) -> None:
        for model in self.models:
            model.fit(x, y)

    @classmethod
    def initialize(
        cls,
        input_size: int,
        *,
        models_count: int,
        seed: Optional[int] = None,
    ) -> EDRVFL:
        random.seed(seed)

        models = [
            DRVFL.initialize(
                input_size,
                seed=random.randint(0, models_count),
                hidden_sizes=[
                    random.randint(32, 64)
                    for _ in range(random.randint(1, 10))
                ],
            ) for _ in range(models_count)
        ]
        return cls(models)


class Evaluate(TypedDict):
    val: Metrics
    test: Metrics


def evaluate(*, train: Path, test: Path, model_init: Callable[[int], Model]) -> Evaluate:
    train_val = pandas.read_csv(train)
    train_val.drop("time", axis=1, inplace=True)

    rmse: List[float] = []
    mape: List[float] = []
    r2: List[float] = []

    def train_and_validate(context_size: int) -> Tuple[StandardScaler, Model, Metrics]:
        train_val_input, train_val_output = extract_xy(train_val, context_size=context_size)
        train_input, val_input, train_output, val_output = train_test_split(
            train_val_input,
            train_val_output,
            train_size=0.8,
            random_state=42,
            shuffle=True,
        )

        scaler = StandardScaler()
        train_input_scaled = scaler.fit_transform(train_input)
        val_input_scaled = scaler.transform(val_input)

        model = model_init(train_input_scaled.shape[1])
        model.fit(train_input_scaled, train_output)
        return scaler, model, metrics(val_output, model.predict(val_input_scaled))

    for context_size in range(1, 21):
        _, _, m = train_and_validate(context_size)
        rmse.append(m["rmse"])
        mape.append(m["mape"])
        r2.append(m["r2"])

    context_size = rank_metrics(
        range(1, 21),
        lambda c: rmse[c - 1],
        lambda c: mape[c - 1],
        lambda c: 1 - r2[c - 1],
    )
    scaler, model, val_metrics = train_and_validate(context_size)

    test_data = pandas.read_csv(test)
    test_data.drop("time", axis=1, inplace=True)
    test_input, test_output = extract_xy(test_data, context_size=context_size)

    test_input_scaled = scaler.transform(test_input)
    test_metrics = metrics(test_output, model.predict(test_input_scaled))

    return Evaluate(val=val_metrics, test=test_metrics)


ROOT = Path(__file__).parent.parent.resolve()
vn30 = ROOT / "data" / "vn30"
outputs = ROOT / "outputs"
outputs.mkdir(parents=True, exist_ok=True)


def test(*, output: Path, model_init: Callable[[int], Model]) -> None:
    with output.open("w", encoding="utf-8", buffering=1) as writer:
        writer.write("sep=,\n")
        writer.write("Train,Test,[val]RMSE,[val]MAPE,[val]R2,[test]RMSE,[test]MAPE,[test]R2\n")

        train_paths = sorted(vn30.glob("*_train.csv"))
        test_paths = sorted(vn30.glob("*_test.csv"))
        for train, test in zip(train_paths, test_paths, strict=True):
            edrvfl = evaluate(
                train=train,
                test=test,
                model_init=model_init,
            )
            writer.write(
                ",".join([
                    train.stem,
                    test.stem,
                    str(edrvfl["val"]["rmse"]),
                    str(edrvfl["val"]["mape"]),
                    str(edrvfl["val"]["r2"]),
                    str(edrvfl["test"]["rmse"]),
                    str(edrvfl["test"]["mape"]),
                    str(edrvfl["test"]["r2"]),
                ])
            )
            writer.write("\n")


test(
    output=outputs / "rvfl.csv",
    model_init=lambda input_size: DRVFL.initialize(
        input_size,
        seed=42,
        hidden_sizes=(32,),
    ),
)
test(
    output=outputs / "drvfl.csv",
    model_init=lambda input_size: DRVFL.initialize(
        input_size,
        seed=42,
        hidden_sizes=(128, 64, 32),
    ),
)
test(
    output=outputs / "edrvfl.csv",
    model_init=lambda input_size: EDRVFL.initialize(
        input_size,
        models_count=8,
        seed=42,
    ),
)
