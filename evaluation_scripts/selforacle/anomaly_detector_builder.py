from .anomaly_detector import AnomalyDetector
from .single.autoencoders.convolutional_autoencoder import ConvolutionalAutoencoder
from .single.autoencoders.deep_autoencoder import DeepAutoencoder
from .single.autoencoders.simple_autoencoder import SimpleAutoencoder
from .single.autoencoders.variational_autoencoder import VariationalAutoencoder

MODELS = {
    'CAE': lambda **kwargs: ConvolutionalAutoencoder(**kwargs),
    'DAE': lambda **kwargs: DeepAutoencoder(**kwargs),
    'SAE': lambda **kwargs: SimpleAutoencoder(**kwargs),
    'VAE': lambda **kwargs: VariationalAutoencoder(**kwargs),
}

def init_anomaly_detector(model_type, name, **kwargs) -> AnomalyDetector:
    builder = MODELS.get(model_type)
    if builder is None:
        raise Exception(f'Unknown model type: "{model_type}"')
    return builder(name=name, **kwargs)
