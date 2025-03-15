"""Flask application package."""

import os
import secrets


def create_app(config=None):
    """Create and configure the Flask application."""
    from flask import Flask

    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        MODEL_DIR=None,
        DATA_DIR=None,
    )
    if config:
        app.config.update(config)

    from app.model_utils import DATA_DIR, MODEL_DIR, load_artifacts
    from app.routes import bp

    artifact_dir = app.config["MODEL_DIR"] or MODEL_DIR
    app.config["MODEL_DIR"] = artifact_dir
    app.config["DATA_DIR"] = app.config["DATA_DIR"] or DATA_DIR
    app.extensions["nids_artifacts"] = load_artifacts(artifact_dir)
    app.register_blueprint(bp)
    return app
