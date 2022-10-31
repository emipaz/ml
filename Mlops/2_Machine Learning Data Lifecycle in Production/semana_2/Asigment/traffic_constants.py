
# Características que deben ser escaladas a la puntuación z
DENSE_FLOAT_FEATURE_KEYS = ['temp', 'snow_1h']

# Características para bucear
BUCKET_FEATURE_KEYS = ['rain_1h']

# Número de cubos utilizados por tf.transform para codificar cada característica.
FEATURE_BUCKET_COUNT = {'rain_1h': 3}

# Característica para escalar de 0 a 1
RANGE_FEATURE_KEYS = ['clouds_all']

# Número de términos de vocabulario utilizados para codificar VOCAB_FEATURES por tf.transform
VOCAB_SIZE = 1000

# Recuento de los cubos fuera de la cabina en los que se han clasificado las VOCAB_FEATURES no reconocidas.
OOV_SIZE = 10

# Características con tipos de datos de cadena que se convertirán en índices
VOCAB_FEATURE_KEYS = [
    'holiday',
    'weather_main',
    'weather_description'
]

# Características con tipo de datos int que se mantendrán tal cual
CATEGORICAL_FEATURE_KEYS = [
    'hour', 'day', 'day_of_week', 'month'
]

# Característica a predecir
VOLUME_KEY = 'traffic_volume'

def transformed_name(key):
    return key + '_xf'
