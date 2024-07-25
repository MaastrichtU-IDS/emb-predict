import sys

from trapi_predict_kit import PredictInput, PredictOutput, trapi_predict


# Define additional metadata to integrate this function in TRAPI
@trapi_predict(
    path="/predict",
    name="Get predicted targets for a given entity",
    description="Return the predicted targets for a given entity: drug (DrugBank ID) or disease (OMIM ID), with confidence scores.",
    # Define which edges can be predicted by this function in a TRAPI query
    edges=[
        {
            "subject": "biolink:Drug",
            "predicate": "biolink:treats",
            "object": "biolink:Disease",
        },
        {
            "subject": "biolink:Disease",
            "predicate": "biolink:treated_by",
            "object": "biolink:Drug",
        },
    ],
    nodes={
        "biolink:Drug": {"id_prefixes": ["DRUGBANK"]},
        "biolink:Disease": {"id_prefixes": ["OMIM"]},
    },
)
def get_predictions(request: PredictInput) -> PredictOutput:
    # Available props: request.subjects, request.objects, request.options

    # Load previously stored models
    # from trapi_predict_kit import load
    # loaded_model = load("models/emb_predict")
    # print(loaded_model.model)

    # Generate predicted associations for the provided input
    # loaded_model.model.predict_proba(x)

    predictions = []
    for subj in request.subjects:
        predictions.append(
            {
                "subject": subj,
                "object": "OMIM:246300",
                "score": 0.12345,
                "object_label": "Leipirudin",
                "object_type": "biolink:Drug",
            }
        )
    for obj in request.objects:
        predictions.append(
            {
                "subject": "DRUGBANK:DB00001",
                "object": obj,
                "score": 0.12345,
                "object_label": "Leipirudin",
                "object_type": "biolink:Drug",
            }
        )
    return {"hits": predictions, "count": len(predictions)}


if __name__ == "__main__":
    # To be run when the script is executed directly
    drug_id = "drugbank:DB00002"
    if len(sys.argv) > 1:
        drug_id = sys.argv[1]
    pi = PredictInput()
    pi.subjects = [drug_id]
    print(get_predictions(pi))
