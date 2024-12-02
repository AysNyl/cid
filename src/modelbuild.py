from dvclive import Live
import yaml
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd

epochs = 10


df = pd.read_csv('..\\data\\train.csv')
print(df)
# clf = LogisticRegressionCV

def main():
    with Live() as live:
        live.log_param("epochs", epochs)
