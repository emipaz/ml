
import tensorflow as tf
import tensorflow_transform as tft

# Keys
_LABEL_KEY = 'label'
_IMAGE_KEY = 'image_raw'


def _transformed_name(key):
    return key + '_xf'

def _image_parser(image_str):
    '''converts the images to a float tensor'''
    image = tf.image.decode_image(image_str, channels=3)
    image = tf.reshape(image, (32, 32, 3))
    image = tf.cast(image, tf.float32)
    return image


def _label_parser(label_id):
    '''one hot encodes the labels'''
    label = tf.one_hot(label_id, 10)
    return label


def preprocessing_fn(inputs):
    """Función de retorno de tf.transform para el preprocesamiento de entradas.
    Args:
        inputs: mapa de claves de características a características crudas aún no transformadas.
    Devuelve:
        Mapa de claves de características de cadena a operaciones de características transformadas.
    """
    
    # Convertir la imagen en bruto y las etiquetas en una matriz de floats y
    # etiquetas codificadas en un punto, respectivamente.
    with tf.device("/cpu:0"):
        outputs = {
            _transformed_name(_IMAGE_KEY):
                tf.map_fn(
                    _image_parser,
                    tf.squeeze(inputs[_IMAGE_KEY], axis=1),
                    dtype=tf.float32),
            _transformed_name(_LABEL_KEY):
                tf.map_fn(
                    _label_parser,
                    tf.squeeze(inputs[_LABEL_KEY], axis=1),
                    dtype=tf.float32)
        }
    
    # escalar los píxeles de 0 a 1
    outputs[_transformed_name(_IMAGE_KEY)] = tft.scale_to_0_1(outputs[_transformed_name(_IMAGE_KEY)])
    
    return outputs
