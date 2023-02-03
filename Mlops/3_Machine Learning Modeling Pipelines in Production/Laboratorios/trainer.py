
from tensorflow import keras
from typing import NamedTuple, Dict, Text, Any, List
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
import tensorflow as tf
import tensorflow_transform as tft

# Define the label key
LABEL_KEY = 'label_xf'

def _gzip_reader_fn(filenames):
  '''Cargar un conjunto de datos comprimidos
  
  Args:
    filenames - nombres de archivos de TFRecords a cargar

  Devuelve:
    TFRecordDataset cargado a partir de los nombres de archivo
  '''

  # Cargar el conjunto de datos. Especifica el tipo de compresión ya que se guarda como `.gz`.
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
  

def _input_fn(file_pattern,
              tf_transform_output,
              num_epochs=None,
              batch_size=32) -> tf.data.Dataset:
  '''Crear lotes de características y etiquetas a partir de registros TF

  Args:
    file_pattern - Lista de archivos o patrones de rutas de archivos que contienen registros de ejemplo.
    tf_transform_output - Gráfico de salida de la transformación
    num_epochs - Número entero que especifica el número de veces que hay que leer el conjunto de datos. 
            Si es None, se recorre el conjunto de datos para siempre.
    batch_size - Un int que representa el número de registros a combinar en un solo lote.

  Devuelve:
    Un conjunto de datos de elementos dict, (o una tupla de elementos dict y etiqueta). 
    Cada dict asigna claves de características a objetos Tensor o SparseTensor.
  '''
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())
  
  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      num_epochs=num_epochs,
      label_key=LABEL_KEY)
  
  return dataset


def model_builder(hp):
  '''
  Construye el modelo y establece los hiperparámetros a afinar.

  Args:
    hp - objeto Keras tuner

  Devuelve:
    Modelo con los hiperparámetros a sintonizar
  '''

  # Inicializar la API secuencial y empezar a apilar las capas
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28, 1)))

  # Get the number of units from the Tuner results
  hp_units = hp.get('units')
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))

  # Add next layers
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(10, activation='softmax'))

  # Get the learning rate from the Tuner results
  hp_learning_rate = hp.get('learning_rate')

  # Setup model for training
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

  # Print the model summary
  model.summary()
  
  return model


def run_fn(fn_args: FnArgs) -> None:
  """Define y entrena el modelo.
  Argumentos:
    fn_args: Contiene los argumentos como pares nombre/valor. Consulte aquí los atributos completos: 
    https://www.tensorflow.org/tfx/api_docs/python/tfx/components/trainer/fn_args_utils/FnArgs#attributes
  """

  # Callback for TensorBoard
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')
  
  # Load transform output
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
  
  # Create batches of data good for 10 epochs
  train_set = _input_fn(fn_args.train_files[0], tf_transform_output, 10)
  val_set = _input_fn(fn_args.eval_files[0], tf_transform_output, 10)

  # Load best hyperparameters
  hp = fn_args.hyperparameters.get('values')

  # Build the model
  model = model_builder(hp)

  # Train the model
  model.fit(
      x=train_set,
      validation_data=val_set,
      callbacks=[tensorboard_callback]
      )
  
  # Save the model
  model.save(fn_args.serving_model_dir, save_format='tf')
