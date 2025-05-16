from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Protocol, Tuple, TYPE_CHECKING

import numpy
from sklearn.linear_model import RidgeCV


ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(ROOT / "models"))


from preprocess import VN30, preprocess_vn30  # type: ignore  # noqa
from RVFL.utils import Metrics, metrics  # type: ignore  # noqa


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


def evaluate(*, symbol: str, model_init: Callable[[int], Model]) -> Metrics:
    train_input, train_output, _, _, test_input, test_output = preprocess_vn30(symbol, val=0)
    model = model_init(train_input.shape[1])
    model.fit(train_input, train_output)
    return metrics(test_output, model.predict(test_input))


def test(*, output: Path, model_init: Callable[[int], Model]) -> None:
    with output.open("w", encoding="utf-8", buffering=1) as writer:
        writer.write("sep=,\n")
        writer.write("Symbol,[test] RMSE,[test] MAPE,[test] R2\n")

        for symbol in VN30:
            result = evaluate(symbol=symbol, model_init=model_init)
            writer.write(
                ",".join([
                    symbol,
                    str(result["rmse"]),
                    str(result["mape"]),
                    str(result["r2"]),
                ])
            )
            writer.write("\n")


outputs = ROOT / "outputs"
outputs.mkdir(parents=True, exist_ok=True)
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
