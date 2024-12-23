import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X, y = make_classification()

clf = joblib.load('models/LRmodel.pkl')

test = pd.read_csv(r"C:\Users\Ayush\cid\data\test.csv")

X = test.drop(columns=['species'])
y = test['species']


y_pred = clf.predict(X)

print(accuracy_score(y, y_pred))
