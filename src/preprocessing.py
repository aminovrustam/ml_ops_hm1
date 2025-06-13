import numpy as np
import pandas as pd
import logging
from sklearn.impute import SimpleImputer 
from category_encoders.cat_boost import CatBoostEncoder
import category_encoders as ce

logger = logging.getLogger(__name__)

monthes = {
    1: 'jan',
    2: 'f',
    3: 'm',
    4: 'a',
    5: 'my',
    6: 'in',
    7: 'j',
    8: 'av',
    9: 's',
    10: 'o',
    11: 'n',
    12: 'd'
}

def w_month(month):
  return monthes[month]

NY_coord = [40.7143, -74.006]
DA_coord = [32.7831, -96.8067]
ORL_coord = [32.7831, -96.8067]
DV_coord = [41.5236, -90.5776]

def day_night(hour):
  if hour >= 6 and hour < 18:
    return 'D'
  else:
    return 'N'
  
def is_weekend(day_of_week):
    return 'W' if day_of_week >= 5 else 'NW'

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float, n_digits: int = 0) -> float:
    """
        Функция для расчёта расстояния от точки А до Б по прямой

        :param lat1: Широта точки А
        :param lon1: Долгота точки А
        :param lat2: Широта точки Б
        :param lon2: Долгота точки Б
        :param n_digits: Округляем полученный ответ до n знака после запятой
        :return: Дистанция по прямой с точностью до n_digits
    """

    lat1, lon1, lat2, lon2 = round(lat1, 6), round(lon1, 6), round(lat2, 6), round(lon2, 6)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)

    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2

    return round(2 * 6372800 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)), n_digits)  # метры.сантиметры


def bearing_degree(lat1: float, lon1: float, lat2: float, lon2: float, n_digits: int = 0) -> float:
    """
        Функция для расчёта угла между прямой [((lat1, lon1), (lat2, lon2)), (нулевой мередиан)]

        :param lat1: Широта точки А
        :param lon1: Долгота точки А
        :param lat2: Широта точки Б
        :param lon2: Долгота точки Б
        :param n_digits: Округляем полученный ответ до n знака после запятой
        :return: Значение угла с точностью до n_digits
    """

    lat1, lon1 = np.radians(round(lat1, 6)), np.radians(round(lon1, 6))
    lat2, lon2 = np.radians(round(lat2, 6)), np.radians(round(lon2, 6))

    dlon = (lon2 - lon1)
    numerator = np.sin(dlon) * np.cos(lat2)
    denominator = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))

    theta = np.arctan2(numerator, denominator)
    theta_deg = (np.degrees(theta) + 360) % 360

    return round(theta_deg, n_digits)

def geo_prep(df):
    logger.debug('Adding geo features...')

    df['bearing_degree_1'] = bearing_degree(df['lat'], df['lon'], df['merchant_lat'], df['merchant_lon'], ).values
    

    df['bearing_degree_2'] = bearing_degree(df['lat'], df['lon'], 0, 0, ).values
    
    df['bearing_degree_3'] = bearing_degree(0, 0, df['merchant_lat'], df['merchant_lon'], ).values
    

    df['bearing_degree_NY_1'] = bearing_degree(df['lat'], df['lon'], NY_coord[0], NY_coord[1], ).values

    df['bearing_degree_NY_2'] = bearing_degree(NY_coord[0], NY_coord[1], df['merchant_lat'], df['merchant_lon'], ).values

    df['bearing_degree_DA_1'] = bearing_degree(df['lat'], df['lon'], DA_coord[0], DA_coord[1], ).values

    df['bearing_degree_DA_2'] = bearing_degree(DA_coord[0], DA_coord[1], df['merchant_lat'], df['merchant_lon'], ).values

    df['bearing_degree_ORL_1'] = bearing_degree(df['lat'], df['lon'], ORL_coord[0], ORL_coord[1], ).values

    df['bearing_degree_ORL_2'] = bearing_degree(ORL_coord[0], ORL_coord[1], df['merchant_lat'], df['merchant_lon'], ).values

    df['bearing_degree_DV_1'] = bearing_degree(df['lat'], df['lon'], DV_coord[0], DV_coord[1], ).values

    df['bearing_degree_DV_2'] = bearing_degree(DV_coord[0], DV_coord[1], df['merchant_lat'], df['merchant_lon'], ).values

    df['hav_dist_1'] = haversine_distance(df['lat'], df['lon'], df['merchant_lat'], df['merchant_lon'], ).values

    df['hav_dist_2'] = haversine_distance(df['lat'], df['lon'], 0, 0, ).values

    df['hav_dist_3'] = haversine_distance(0, 0, df['merchant_lat'], df['merchant_lon'], ).values

    df['hav_dist_NY_1'] = haversine_distance(df['lat'], df['lon'], NY_coord[0], NY_coord[1], ).values

    df['hav_dist_NY_2'] = haversine_distance(NY_coord[0], NY_coord[1], df['merchant_lat'], df['merchant_lon'], ).values

    df['hav_dist_DA_1'] = haversine_distance(df['lat'], df['lon'], DA_coord[0], DA_coord[1], ).values

    df['hav_dist_DA_2'] = haversine_distance(DA_coord[0], DA_coord[1], df['merchant_lat'], df['merchant_lon'], ).values

    df['hav_dist_ORL_1'] = haversine_distance(df['lat'], df['lon'], ORL_coord[0], ORL_coord[1], ).values

    df['hav_dist_ORL_2'] = haversine_distance(ORL_coord[0], ORL_coord[1], df['merchant_lat'], df['merchant_lon'], ).values

    df['hav_dist_DV_1'] = haversine_distance(df['lat'], df['lon'], DV_coord[0], DV_coord[1], ).values

    df['hav_dist_DV_2'] = haversine_distance(DV_coord[0], DV_coord[1], df['merchant_lat'], df['merchant_lon'], ).values
    return df

def time_prep(df):
    logger.debug('Adding time features...')
    df['time'] = pd.to_datetime(df['transaction_time'])

    df['times_of_day'] = df['time'].dt.hour.apply(day_night)

    df['weekends'] = df['time'].dt.dayofweek.apply(is_weekend)

    df['month'] = df['time'].dt.month.apply(w_month)

    df['time'] = df['time'].astype(int) / 10**9

    df = df.drop(columns=['transaction_time'])
    return df


def load_train_data():

    logger.info('Loading training data...')

    # Import Train dataset
    train = pd.read_csv('./train_data/train.csv', sep=',')


    # Define column types
    target_col = 'target'
    categorical_cols = list(train.drop(columns=['target']).select_dtypes([object]).columns)
    continuous_cols = list(train.drop(columns=['target']).select_dtypes([int, float]).columns)
    n_cats = 50

    
    
    logger.info('Raw train data imported. Shape: %s', train.shape)

    # Add some simple time features
    train = time_prep(train)
    
    # Calculate distance between a client and a merchant
    train = geo_prep(train)

    

    logger.info('Train data processed. Shape: %s', train.shape)

    return train

def load_train_data_with_enc():

    logger.info('Loading training data...')

    # Import Train dataset
    train = pd.read_csv('./train_data/train.csv', sep=',')

    # Define column types
    target_col = 'target'
    
    continuous_cols = list(train.drop(columns=['target']).select_dtypes([int, float]).columns)
    n_cats = 50

    logger.info('Raw train data imported. Shape: %s', train.shape)

    # Add some simple time features
    train = time_prep(train)
    
    # Calculate distance between a client and a merchant
    train = geo_prep(train)

    # Cat features
    categorical_cols = list(train.drop(columns=['target']).select_dtypes([object]).columns)
    cbe_encoder = ce.cat_boost.CatBoostEncoder()
    cbe_encoder.fit(train[categorical_cols], train[target_col])

    logger.info('Train data processed. Shape: %s', train.shape)

    # Return both the processed data and the encoder
    return train, cbe_encoder

def run_preproc(train, input_df, encoder):

    # Define column types
    target_col = 'target'
    
   
    
    # Run category encoding
    
    
    # Add some simple time features
    input_df = time_prep(input_df)


    logger.info('Added time features. Output shape: %s', input_df.shape)

    
    # Calculate distance between a client and a merchant
    input_df = geo_prep(input_df)
    
    continuous_cols = list(input_df.select_dtypes([int, float]).columns)
    # Impute empty values with mean value
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
    imputer = imputer.fit(train[continuous_cols])

    categorical_cols = list(input_df.select_dtypes([object]).columns)
    x = encoder.transform(input_df[categorical_cols])
    for column in categorical_cols:
       input_df[column]=x[column]

    output_df = input_df
    
        
    logger.info('Continuous features preprocessing completed. Output shape: %s', output_df.shape)
    
    # Return resulting dataset
    return output_df
