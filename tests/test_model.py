from emb_predict.predict import get_predictions

# from emb_predict.train import train_ot_models
# from emb_application import set_args_from_config
from trapi_predict_kit import PredictInput

input_id = "CHEMBL_4559134"


def test_get_predictions():
    pi = PredictInput()
    pi.subjects = [input_id]
    predictions = get_predictions(pi)
    print(predictions)
    assert len(predictions["hits"]) > 0
    assert len(predictions["hits"]) == predictions["count"]


# def test_train_model():
#     loaded_model = train_ot_models()
#     assert loaded_model is not None
#     # scores = loaded_model.scores
# assert 0.80 < scores['precision'] < 0.95
# assert 0.60 < scores['recall'] < 0.85
# assert 0.80 < scores['accuracy'] < 0.95
# assert 0.85 < scores['roc_auc'] < 0.95
# assert 0.70 < scores['f1'] < 0.85
# assert 0.75 < scores['average_precision'] < 0.95
