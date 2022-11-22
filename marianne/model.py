"""The model manager for marianne"""
# marianne/model.py

import os

import click
import joblib
import polars as pl
from flask import current_app, g
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


def get_model():
    if "model" not in g or "vc" not in g:
        g.model = joblib.load(
            os.path.join(current_app.config["SPAM_DETECT_MODEL"], "randomforest.model")
        )
        g.vc = joblib.load(
            os.path.join(current_app.config["SPAM_DETECT_MODEL"], "randomforest.vc")
        )

    return [g.model, g.vc]


def close_model(e=None):
    g.pop("model", None)
    g.pop("vc", None)


def predict_text(text):
    [model, vc] = get_model()
    cv_text = vc.transform([text])
    return model.predict(cv_text)


def init_model():
    data = pl.read_csv(
        os.path.join(current_app.config["DATA_PATH"], "spam_and_ham_text.csv")
    )
    label = data.select(
        [
            pl.col("label"),
        ]
    )
    text = data.select(
        [
            pl.col("text"),
        ]
    )
    vc = CountVectorizer()
    x_train_counts = vc.fit_transform(text.to_numpy()[0])
    model = RandomForestClassifier(n_estimators=100)
    model.fit(x_train_counts, label.to_numpy()[0])
    joblib.dump(
        model, current_app.config["SPAM_DETECT_MODEL"] + "/" + "randomforest.model"
    )
    joblib.dump(vc, current_app.config["SPAM_DETECT_MODEL"] + "/" + "randomforest.vc")


@click.command("init-model")
def init_model_command():
    """Clear the existing model and create new model."""
    init_model()
    click.echo("Initialized the model.")


def init_app(app):
    app.teardown_appcontext(close_model)
    app.cli.add_command(init_model_command)
