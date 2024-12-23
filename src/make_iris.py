import pathlib
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def spldata(data):
    df = pd.read_csv(data)
    return train_test_split(df, random_state=42)


def main():
    try:
        data = pathlib.Path(sys.argv[1])
        print(data.parent.is_dir())
        if data.parent.is_dir():
            a, b = spldata(data)
            model = pathlib.Path("./models/data")
            pathlib.Path(model).mkdir(exist_ok=True)
            a.to_csv(model.as_posix() + "/train.csv", index=False)
            b.to_csv(model.as_posix() + "/test.csv", index=False)
        else:
            raise Exception("Invalid path")
    except Exception:
        print(Exception)


if __name__ == "__main__":
    main()
