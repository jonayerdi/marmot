import numpy as np
from keras.utils import Sequence

from utils.images import get_images, load_image

class ImagesBatchGenerator(Sequence):
    """
    Single image based batch generator. Generated inputs == generated labels (i.e., x == y) as required by autoencoders
    """
    
    def __init__(self, data_dir, batch_size, img_shape, img_processing):
        self.image_paths = list(get_images(data_dir))
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.img_processing = img_processing

    def load_images(self, image_paths):
        X = np.empty(shape=(len(image_paths),) + (self.img_shape))
        i = 0
        for image_path in image_paths:
            X[i,:] = load_image(image_path=image_path, processing=self.img_processing)
            i += 1
        return X

    def __getitem__(self, index):
        X = self.load_images(self.image_paths[index*self.batch_size:(index+1)*self.batch_size])
        return X, X

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def get_batch_size(self):
        return self.batch_size
