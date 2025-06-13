import pandas as pd
import logging
from catboost import CatBoostClassifier
import json

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
model = CatBoostClassifier()
model.load_model('./models/cbClassifier.cbm')

# Define optimal threshold
# model_th = 0.98
logger.info('Pretrained model imported successfully...')

# Make prediction
def make_pred(dt, path_to_file):
    # Make submission dataframe
    submission = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': (model.predict(dt))
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    # Return proba for positive class
    return submission

def get_feature_import(path_to_file):
    feature_importances = model.get_feature_importance()
    feature_names = model.feature_names_

    # Создание словаря с важностями
    importance_dict = dict(zip(feature_names, feature_importances))

    # Сортировка и получение топ-5 фич
    top_5_features = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])

    return top_5_features

def get_probability(dt):
    proba = model.predict_proba(dt).T[1]
    return proba