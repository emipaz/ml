
import tensorflow as tf
import tensorflow_transform as tft

import traffic_constants

# Unpack the contents of the constants module
_DENSE_FLOAT_FEATURE_KEYS = traffic_constants.DENSE_FLOAT_FEATURE_KEYS
_RANGE_FEATURE_KEYS       = traffic_constants.RANGE_FEATURE_KEYS
_VOCAB_FEATURE_KEYS       = traffic_constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE               = traffic_constants.VOCAB_SIZE
_OOV_SIZE                 = traffic_constants.OOV_SIZE
_CATEGORICAL_FEATURE_KEYS = traffic_constants.CATEGORICAL_FEATURE_KEYS
_BUCKET_FEATURE_KEYS      = traffic_constants.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT     = traffic_constants.FEATURE_BUCKET_COUNT
_VOLUME_KEY               = traffic_constants.VOLUME_KEY
_transformed_name         = traffic_constants.transformed_name


def preprocessing_fn(inputs):
    """Función de retorno de tf.transform para el preprocesamiento de entradas.
    Args:
    inputs: mapa de claves de características a características crudas aún no transformadas.
    Devuelve:
    Mapa de claves de rasgos en cadena a operaciones de rasgos transformados.
    """
    outputs = {}

    ### START CODE HERE
    
    # Escala estas características a la puntuación z.
    for key in _DENSE_FLOAT_FEATURE_KEYS:
        # Escala estas características a la puntuación z
        outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])

            

    # Scalificar estas características de 0 a 1
    for key in _RANGE_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(inputs[key])
            

    # Transformar las cadenas en índices 
    # Sugerencia: utilice VOCAB_SIZE y OOV_SIZE para definir los parámetros top_k y num_oov
    for key in _VOCAB_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
                                                        inputs[key],
                                                        top_k           = _VOCAB_SIZE,
                                                        num_oov_buckets = _OOV_SIZE)
            
            
            

    # Bucketize the feature
    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(
                                                    x = inputs[key], 
                                                    num_buckets = _FEATURE_BUCKET_COUNT[key])
                                                    
            

    # Mantenga las características tal y como están. No se necesita la función tft.
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = inputs[key]

        
    # Utilice `tf.cast` para convertir la clave de la etiqueta en float32
    traffic_volume = tf.cast(inputs[_VOLUME_KEY], tf.float32)
  
    
    # Crear una característica que muestre si el volumen de tráfico es mayor que la media y convertirlo en un int
    outputs[_transformed_name(_VOLUME_KEY)] = tf.cast(  
        
        # Utilice `tf.greater` para comprobar si el volumen de tráfico de una fila es mayor 
        # que la media de toda la columna de volumen de tráfico
        tf.greater(traffic_volume, tft.mean(tf.cast(traffic_volume, tf.float32))),
        
        tf.int64)                                        

    ### END CODE HERE
    return outputs
