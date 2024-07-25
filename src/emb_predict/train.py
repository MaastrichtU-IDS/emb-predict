import logging

import typer
from trapi_predict_kit import save
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

log = logging.getLogger()

cli = typer.Typer(help="Training for models")


def load_data():
    data, y = load_iris(return_X_y=True, as_frame=True)
    return data, y


def fit_classifier(hyper_params, data, y):
    clf = RandomForestClassifier(
        n_jobs=hyper_params["n_jobs"],
        random_state=hyper_params["random_state"],
    )
    clf.fit(data, y)
    return clf


def evaluate(model):
    # Evaluate the quality of your model using custom metrics
    # cf. https://scikit-learn.org/stable/modules/model_evaluation.html
    return {
        "precision": 0.85,
        "recall": 0.80,
        "accuracy": 0.85,
        "roc_auc": 0.90,
        "f1": 0.75,
        "average_precision": 0.85,
    }


def save_model(model, path, sample_data, scores, hyper_params):
    loaded_model = save(
        model,
        path,
        sample_data=sample_data,
        scores=scores,
        hyper_params=hyper_params,
    )
    return loaded_model


@cli.command(help="Train a model")
def training_workflow(n_jobs: int = 2):
    # Define models hyper params
    hyper_params = {"n_jobs": n_jobs, "random_state": 42}

    data, y = load_data()

    # Train model (here we use a stub dataset just to make the example clear)
    model = fit_classifier(hyper_params, data, y)

    # Evaluate the model using your own metrics
    scores = evaluate(model)

    # Save the model generated to the models/ folder
    loaded_model = save_model(
        model,
        "models/emb_predict",
        sample_data=data,
        scores=scores,
        hyper_params=hyper_params,
    )
    return loaded_model
    # Optionally you can save other files (dataframes, JSON objects) in the data/ folder


if __name__ == "__main__":
    cli()
