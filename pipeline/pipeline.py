from typing import Callable, List

import pandas as pd


def run_pipeline(steps: List[Callable]):
    data: dict[str, pd.DataFrame] = {}

    for step in steps:
        print(step.__name__)
        step(data)
