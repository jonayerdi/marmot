from ..autoencoder import Autoencoder
from utils.images import SO_IMAGE_HEIGHT, SO_IMAGE_WIDTH, SO_IMAGE_CHANNELS

INPUT_SHAPE = (SO_IMAGE_HEIGHT * SO_IMAGE_WIDTH * SO_IMAGE_CHANNELS,)

class DeepAutoencoder(Autoencoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_shape(self):
        return INPUT_SHAPE

    def init_model(self):
        from keras import Sequential
        from keras.layers import Dense

        hidden_layer_dim=256
        model = Sequential()
        model.add(Dense(hidden_layer_dim, activation='relu', input_shape=INPUT_SHAPE))
        model.add(Dense(hidden_layer_dim // 2, activation='relu'))
        model.add(Dense(hidden_layer_dim // 4, activation='sigmoid'))
        model.add(Dense(hidden_layer_dim // 2, activation='relu'))
        model.add(Dense(hidden_layer_dim, activation='relu'))
        model.add(Dense(SO_IMAGE_HEIGHT * SO_IMAGE_WIDTH * SO_IMAGE_CHANNELS, activation='sigmoid'))

        return model

    def normalize_and_reshape(self, img):
        img = img.astype('float32') / 255.
        img = img.reshape(-1, SO_IMAGE_HEIGHT * SO_IMAGE_WIDTH * SO_IMAGE_CHANNELS)
        return img
