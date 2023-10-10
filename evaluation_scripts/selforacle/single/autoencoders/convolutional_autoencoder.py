from ..autoencoder import Autoencoder
from utils.images import SO_IMAGE_HEIGHT, SO_IMAGE_WIDTH, SO_IMAGE_CHANNELS

INPUT_SHAPE = (SO_IMAGE_HEIGHT, SO_IMAGE_WIDTH, SO_IMAGE_CHANNELS)


class ConvolutionalAutoencoder(Autoencoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_shape(self):
        return INPUT_SHAPE

    def init_model(self):
        from keras import Input, Model
        from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation

        input_img = Input(shape=INPUT_SHAPE)

        x = Conv2D(64, (3, 3), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(16, (3, 3), padding='same')(encoded)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        decoded = Activation('sigmoid')(x)

        autoencoder = Model(input_img, decoded)
        return autoencoder

    def normalize_and_reshape(self, img):
        img = img.astype('float32') / 255.
        img = img.reshape(-1, SO_IMAGE_HEIGHT, SO_IMAGE_WIDTH, SO_IMAGE_CHANNELS)  # CNN needs depth.
        return img
