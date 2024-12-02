import pathlib
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

def spldata():
    df = sns.load_dataset('iris')
    return train_test_split(df, random_state=42)

def main():
    try:
        data = pathlib.Path(sys.argv[1])
        print(data.is_dir())
        if data.is_dir():
            a, b = spldata()
            a.to_csv(data.as_posix() + '/train.csv')
            b.to_csv(data.as_posix() + '/test.csv')
        else:
            raise Exception("Invalid path") 
    except Exception:
        print(Exception)


if __name__ == '__main__':
    main()