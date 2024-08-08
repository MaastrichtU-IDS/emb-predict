import os
from emb_predict.utils import log
from trapi_predict_kit.config import settings
from trapi_predict_kit import TRAPI
from fastapi import FastAPI
from contextlib import asynccontextmanager
from emb_predict.application import (
    load_application_components,
    set_project_paths,
)
from emb_predict.predict import get_predictions


# preload all the components of the application
@asynccontextmanager
async def lifespan(app: FastAPI):
    config_file = os.getenv("CONFIG_FILE")
    log.info(f"Loading application components from {config_file}")
    paths = set_project_paths(config_file=config_file)
    pre_loaded_data = load_application_components(paths)
    pre_loaded_data[
        "web_app"
    ] = True  # set to True to indicate that we are running in a web app
    yield


app = TRAPI(
    predict_endpoints=[get_predictions],
    lifespan=lifespan,
    title="Embedding prediction toolkit",
    version="1.0.0",
    openapi_version="3.0.1",
    description="""Machine learning models that uses embeddings to produce drug-disease predictions.
\n\nService supported by the [NCATS Translator project](https://ncats.nih.gov/translator/about)""",
    # itrb_url_prefix="emb-predict",
    # dev_server_url="https://emb-predict.semanticscience.org",
    info={
        "contact": {
            "name": "Michel Dumontier",
            "email": "michel.dumontier@maastrichtuniversity.nl",
            "x-id": "https://orcid.org/0000-0003-4727-9435",
            "x-role": "responsible developer",
        },
        "license": {
            "name": "MIT license",
            "url": "https://opensource.org/licenses/MIT",
        },
        "termsOfService": "https://github.com/micheldumontier/emb-predict/blob/main/LICENSE.txt",
        "x-translator": {
            "component": "KP",
            "team": ["Clinical Data Provider"],
            "biolink-version": settings.BIOLINK_VERSION,
            "infores": "infores:emb_predict",
            "externalDocs": {
                "description": "The values for component and team are restricted according to this external JSON schema. See schema and examples at url",
                "url": "https://github.com/NCATSTranslator/translator_extensions/blob/production/x-translator/",
            },
        },
        "x-trapi": {
            "version": settings.TRAPI_VERSION,
            "asyncquery": False,
            "operations": [
                "lookup",
            ],
            "externalDocs": {
                "description": "The values for version are restricted according to the regex in this external JSON schema. See schema and examples at url",
                "url": "https://github.com/NCATSTranslator/translator_extensions/blob/production/x-trapi/",
            },
        },
    },
    trapi_example={
        "message": {
            "query_graph": {
                "edges": {
                    "e01": {
                        "subject": "n0",
                        "predicates": ["biolink:treats"],
                        "object": "n1",
                    },
                },
                "nodes": {
                    "n0": {
                        "categories": ["biolink:Drug"],
                        "ids": [
                            "CHEMBL_4559134",
                        ],
                    },
                    "n1": {
                        "categories": ["biolink:Disease"],
                        "ids": [
                            "MONDO_0004992",
                        ],
                    },
                },
            }
        },
        "query_options": {"max_score": 1, "min_score": 0.35, "n_results": 10},
    },
)
