import logging

from utils.logging import init_logger
from ..anomaly_detector import AnomalyDetector
from .autoencoder_batch_generator import AutoencoderBatchGenerator

logger = logging.Logger("Autoencoder")
init_logger(logger)

class Autoencoder(AnomalyDetector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_batch_generator(self, data_dir: str, batch_size: int, img_processing='selforacle', **kwargs):
        return AutoencoderBatchGenerator(
            data_dir=data_dir, anomaly_detector=self, 
            img_processing=img_processing, batch_size=batch_size,
        )
