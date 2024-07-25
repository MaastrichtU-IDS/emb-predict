import logging

from trapi_predict_kit.config import settings
from trapi_predict_kit import TRAPI
from emb_predict.predict import get_predictions


log_level = logging.ERROR
DEV_MODE = False
if DEV_MODE is True:
    log_level = logging.INFO
logging.basicConfig(level=log_level)

openapi_info = {
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
        "infores": "infores:openpredict",
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
}

app = TRAPI(
    predict_endpoints=[get_predictions],
    info=openapi_info,
    title="Embedding prediction toolkit",
    version="1.0.0",
    openapi_version="3.0.1",
    description="""Machine learning models that uses embeddings to produce drug-disease predictions.
\n\nService supported by the [NCATS Translator project](https://ncats.nih.gov/translator/about)""",
    # itrb_url_prefix="emb-predict",
    # dev_server_url="https://emb-predict.semanticscience.org",
)
