import abc
import logging
import os
from os.path import join, abspath

import numpy as np

from utils.logging import init_logger
from utils.thresholds import calculate_losses_predictions, euclidean_distance, save_thresholds, THRESHOLD_CONFIDENCE_INTERVALS

MODELS_LOCATION = "models"

logger = logging.Logger("AnomalyDetector")
init_logger(logger)

get_output_prefix = lambda model_name: join(MODELS_LOCATION, f'{model_name}')

class AnomalyDetector(abc.ABC):

    @abc.abstractmethod
    def init_model(self):
        logger.error("method must be overriden in child class")
        exit(1)

    @abc.abstractmethod
    def normalize_and_reshape(self, img: np.array) -> np.array:
        logger.error("method must be overriden in child class")
        exit(1)

    @abc.abstractmethod
    def get_batch_generator(self, data_dir: str, batch_size: int, img_processing, **kwargs):
        logger.error("method must be overriden in child class")
        exit(1)

    def __init__(self, name, init_model=True):
        self.name = name
        self.model = None
        self.tflite_interpreter = None
        if init_model:
            self.model = self.init_model()
            self.compile_model()

    def compile_model(self):
        self.model.summary()
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def get_prefix(self):
        return get_output_prefix(self.name)

    def train(self, batch_generator, epochs: int):
        assert self.model is not None, 'Model has not been initialized'
        self.model.fit(
            x=batch_generator,
            epochs=epochs,
            use_multiprocessing=False,
        )

    def predict_with_model(self, x):
        assert self.model is not None, 'Model has not been initialized'
        return self.model.predict(x=x)

    def predict_with_tflite(self, x):
        assert self.tflite_interpreter is not None, 'tflite interpreter has not been initialized'
        input_details = self.tflite_interpreter.get_input_details()
        output_details = self.tflite_interpreter.get_output_details()
        predictions = []
        for input in x:
            self.tflite_interpreter.set_tensor(input_details[0]["index"], input)
            self.tflite_interpreter.invoke()
            predictions.append(self.tflite_interpreter.get_tensor(output_details[0]["index"]))
        return predictions

    def predict(self, x, mode='model'):
        return {
            'model': self.predict_with_model,
            'tflite': self.predict_with_tflite,
        }[mode](x=x)

    def calculate_losses_inputs(self, inputs, labels, distance_metric=euclidean_distance, mode='model'):
        return calculate_losses_predictions(labels=labels, predictions=self.predict(x=inputs, mode=mode), distance_metric=distance_metric)

    def calculate_losses(self, batch_generator, distance_metric=euclidean_distance, mode='model') -> np.array:
        logger.info("Calculating losses for %s" % self.name)
        count = len(batch_generator) * batch_generator.get_batch_size()
        distances = np.empty(shape=(count,))
        index = 0
        for inputs, labels in batch_generator:
            if index % 10 == 0:
                progress = index*100//count
                logger.info(f"Progress: {progress}%)")
            # Predict with model
            predictions = self.predict(x=inputs, mode=mode)
            assert len(labels) == len(predictions)
            # Calculate distances
            for label, prediction in zip(labels, predictions):
                distance = distance_metric(label, prediction)
                distances.put(index, distance)
                index += 1
        return distances

    def save_thresholds(self, losses, prefix=None, conf_intervals=THRESHOLD_CONFIDENCE_INTERVALS):
        if prefix is None:
            prefix = self.get_prefix()
        save_thresholds(losses=losses, prefix=prefix, conf_intervals=conf_intervals)

    def save_model(self, path=None, format='h5'):
        assert self.model is not None, 'Model has not been initialized'

        if path is None:
            path = f'{self.get_prefix()}.{format}'

        os.makedirs(abspath(join(path, '..')), exist_ok=True)

        if format == 'tflite':
            import tensorflow as tf
            converter = tf.compat.v2.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            with open(path, 'wb') as f:
                f.write(tflite_model)
        else:
            self.model.save(path, save_format=format)

    def load_model(self, path=None, format='h5'):
        
        if path is None:
            path = f'{self.get_prefix()}.{format}'

        if format == 'tflite':
            import tensorflow as tf
            self.tflite_interpreter = tf.lite.Interpreter(model_path=path)
            self.tflite_interpreter.allocate_tensors()
        else:
            assert self.model is not None, 'Model has not been initialized'
            self.model.load_weights(path)
