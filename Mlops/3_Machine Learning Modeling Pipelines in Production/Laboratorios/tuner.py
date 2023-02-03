
# Define imports
from kerastuner.engine import base_tuner
import kerastuner as kt
from tensorflow import keras
from typing import NamedTuple, Dict, Text, Any, List
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
import tensorflow as tf
import tensorflow_transform as tft

# Declare namedtuple field names
TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

# Label key
LABEL_KEY = 'label_xf'

# Callback for the search strategy
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


def _gzip_reader_fn(filenames):
  '''Lconjunto de datos comprimidos oad
  
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

  # Get feature specification based on transform output
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())
  
  # Create batches of features and labels
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
    hp - Objeto Keras tuner

  Devuelve:
    Modelo con los hiperparámetros a sintonizar
  '''

  # Inicializar la API secuencial y empezar a apilar las capas
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28, 1)))

  # Ajuste el número de unidades en la primera capa densa
  # Elija un valor óptimo entre 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(keras.layers.Dense(units=hp_units, activation='relu', name='dense_1'))

  # Add next layers
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(10, activation='softmax'))

  # Ajuste la tasa de aprendizaje para el optimizador
  # Elija un valor óptimo entre 0,01, 0,001 o 0,0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

  return model

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """Construye el sintonizador utilizando la API KerasTuner.
  Argumentos:
    fn_args: contiene los argumentos como pares nombre/valor.

      - working_dir: directorio de trabajo para el ajuste.
      - train_files: Lista de rutas de archivos que contienen datos de entrenamiento tf.Example.
      - eval_files: Lista de rutas de archivos que contienen datos de tf.Example de evaluación.
      - train_steps: número de pasos de entrenamiento.
      - eval_steps: número de pasos de evaluación.
      - schema_path: esquema opcional de los datos de entrada.
      - transform_graph_path: gráfico de transformación opcional producido por TFT.
  
  Devuelve:
    Una namedtuple que contiene lo siguiente:
      - tuner: un BaseTuner que se utilizará para el ajuste.
      - fit_kwargs: Args para pasar a la función run_trial del sintonizador para ajustar el
                    modelo, por ejemplo, el conjunto de datos de entrenamiento y validación. Se requiere
                    depende de la implementación del sintonizador anterior.
  """

  # Define tuner search strategy
  tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory=fn_args.working_dir,
                     project_name='kt_hyperband')

  # Load transform output
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

  # Use _input_fn() to extract input features and labels from the train and val set
  train_set = _input_fn(fn_args.train_files[0], tf_transform_output)
  val_set = _input_fn(fn_args.eval_files[0], tf_transform_output)


  return TunerFnResult(
      tuner=tuner,
      fit_kwargs={ 
          "callbacks":[stop_early],
          'x': train_set,
          'validation_data': val_set,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      }
  )
