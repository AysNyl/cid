import mlflow
from mlflow.models import infer_signature
import pandas as pd
import pathlib
from sklearn import datasets

from sklearn import pipeline
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression


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

opline = pipeline.Pipeline([("Standardization", clt),
                            ('Model_training', clf)])

X = train.drop(columns=['species'])
y = train['species']

opline.fit(X, y)

test = pd.read_csv(RDIR.joinpath('models/data/test.csv'))

pred = opline.predict(test.drop(columns=['species']))

accuracy = accuracy_score(test['species'], pred)

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new mlflow Experiment
mlflow.set_experiment('Mlflow Quickstart')

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params=params)

    # log the loss metric
    mlflow.log_metric('accuracy', accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    mlflow.set_tag('Training Info', 'Basic LR Model for iris dataset')

    # Infer the model signature
    signature = infer_signature(X, opline.predict(X=X))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=opline,
        artifact_path="iris_model",
        signature=signature,
        input_example=X,
        registered_model_name="tracking-quickstart",
    )


# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(test.drop(columns=['species']))

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(test.drop(columns=['species']).to_numpy(), columns=iris_feature_names)
result["actual_class"] = test['species']
result["predicted_class"] = predictions

print(result[:4])