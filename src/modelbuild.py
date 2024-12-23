from dvclive import Live
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlxtend.plotting import plot_decision_regions
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
import joblib
import pathlib
from sklearn.utils.validation import check_is_fitted

# with open(r"C:\Users\Ayush\cid\dvc.yaml", 'r') as file:
#     PARAMS = yaml.safe_load(file)['params']

# for i in PARAMS:
#     with open(i, 'r') as file:
#         NUM_EPOCHS = yaml.safe_load(file)['epochs']


with open("params.yaml", 'r') as fpm:
    params = yaml.safe_load(fpm)['train']

NUM_EPOCHS = params["epochs"]

with Live() as live:
    
    live.log_param("epochs", NUM_EPOCHS)
    
    for epoch in range(NUM_EPOCHS):
        # clf_logistic_regression = LogisticRegression(random_state=42)
        # clf_naive_bayes = GaussianNB()
        # clf_random_forest = RandomForestClassifier(random_state=42)

        # clf_ensemble = EnsembleVoteClassifier(
        #     clfs= [clf_logistic_regression, clf_naive_bayes, clf_random_forest],
        #     weights= [2, 1, 1],
        #     voting='soft'
        # )

        # all_classifiers = [
        #     ("Logistic Regression", clf_logistic_regression),
        #     ("Naive Bayes", clf_naive_bayes),
        #     ("Random Forest", clf_random_forest),
        #     ("Ensemble", clf_ensemble),
        # ]

        train = pd.read_csv(r"C:\Users\Ayush\cid\data\train.csv")

        Clt = ColumnTransformer([('input', StandardScaler(), [2, 3])],
                                 remainder='drop')

        Clr = LogisticRegression(penalty='l2', C=params['c'], max_iter=params['max_iter'])

        opline = Pipeline([('column transformer', Clt),
                           ('Logistic Regression', Clr)])
        
        '''pipelines will always pass y through unchanged. 
        Do the transformation outside the pipeline.
        because the scikit-learn original Pipeline does not change the y 
        or the number of samples in X and y during the steps.
        (This is a known design flaw in scikit-learn, 
        but it's never been pressing enough to change or extend the API.)'''

        X = train.drop(columns=['species'])
        y = train['species']

        # unworkabe with string classification
        mdl = TransformedTargetRegressor(opline, transformer=LabelEncoder())

        metrics = cross_validate(opline, X, y, scoring='accuracy', cv=5)

        for metric_name, value in metrics.items():
            live.log_metric(metric_name, value.mean())


        live.next_step()
    # cross validation is performed only for validation, it doesn't fit your original model
    # it creates its own internal model instances for for model training, checking & validation
    # original model remains untrained & unchanged.
    
    # train the original model.
    opline.fit(X=X, y=y)
    
    # waiting for the TransformedTargetClassifier
    # mdl.fit(X=X, y=y)
    
    pathlib.Path("models").mkdir(exist_ok=True)
    
    joblib.dump(opline, "models/LRmodel.pkl")

    live.log_artifact("models/LRmodel.pkl", type="model")

# fig, ax = plt.subplots(1, 1, figsize=(28, 24))

# le = LabelEncoder()

# print(X)
# plot_decision_regions(
#         X=X.to_numpy(), y=le.fit_transform(y), clf=joblib.load("models/LRmodel.pkl"), 
#         legend=2, ax=ax, feature_index=(2,3), filler_feature_values={0: [0], 1: [0]},
#         filler_feature_ranges={0: 0.5, 1: 0.5}
#     )

# ax.set_title("Logistic Regression", fontsize=12, fontweight="bold")
# ax.tick_params(axis='both', which='major', labelsize=8)
# ax.set_xlabel("petal_length", fontsize=9, fontweight="bold")
# ax.set_ylabel("petal_width", fontsize=9, fontweight="bold")

        # x2d = df[['petal_length', 'petal_width']].to_numpy()
        # le = LabelEncoder()
        # y = le.fit_transform(df['species'])

        # fig, axs = plt.subplots(2, 2, figsize=(28, 24), sharey=True, sharex=True)

        # for classifier, grid in zip(
        #     all_classifiers, product([0, 1], [0, 1])
        # ):
        #     clf_name, clf = classifier[0], classifier[1]
        #     ax = axs[grid[0], grid[1]]

        #     clf.fit(x2d, y)

        #     plot_decision_regions(
        #         X=x2d, y=y, clf=clf, legend=2, ax=ax
        #     )

        #     ax.set_title(clf_name, fontsize=12, fontweight="bold")
        #     ax.tick_params(axis='both', which='major', labelsize=8)
        #     ax.set_xlabel("petal_length", fontsize=9, fontweight="bold")
        #     ax.set_ylabel("petal_width", fontsize=9, fontweight="bold")

        # plt.show()



    # print(df)
    # clf = LogisticRegressionCV

    # def main():
    #     with Live() as live:
    #         live.log_param("epochs", epochs)
