import mlflow
import pathlib
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature


RDIR = pathlib.Path('C:/Users/Ayush/cid')

train = pd.read_csv(RDIR.joinpath('models/data/train.csv'))

clt = ColumnTransformer([('standard scaler', StandardScaler(), ['petal_length', 'petal_width'])], remainder='drop')

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    # "multi_class": "auto",
    "random_state": 8888,
}

clf = LogisticRegression(**params)

opline = Pipeline([("Standardization", clt),
                   ('Model_training', clf)])

X = train.drop(columns=['species'])
y = train['species']

opline.fit(X, y)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

signature = infer_signature(X, opline.predict(X=X))

model_info = mlflow.sklearn.log_model(
    sk_model=opline,
    artifact_path="iris_model",
    signature=signature,
    input_example=X,
    registered_model_name="tracking-quickstart",
)

test = pd.read_csv(RDIR.joinpath('models/data/test.csv'))

# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(test.drop(columns=['species']))

iris_feature_names = sns.datasets.load_iris().feature_names

result = pd.DataFrame(test.drop(columns=['species']), columns=iris_feature_names)
result["actual_class"] = test['species']
result["predicted_class"] = predictions

result[:4]