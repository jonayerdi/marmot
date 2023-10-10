import abc

import cv2
import numpy as np

from utils.images import process_image
from utils.images_batch_generator import ImagesBatchGenerator

class MR(abc.ABC):
    def __init__(self):
        super().__init__()
    def __str__(self) -> str:
        return self.__class__.__name__
    @abc.abstractmethod
    def transform_input(self, input):
        raise Exception()
    @abc.abstractmethod
    def expected_output(self, output):
        raise Exception()
    def get_followup_batch_generator(self, data_dir, batch_size, img_shape, img_processing):
        return ImagesBatchGenerator(
            data_dir=data_dir,
            batch_size=batch_size,
            img_shape=img_shape,
            img_processing=lambda image, *args: process_image(
                image=cv2.cvtColor(
                    src=self.transform_input(
                        process_image(image=image, processing='rgb')
                    ),
                    code=cv2.COLOR_RGB2BGR,
                ), 
                processing=img_processing,
            )
        )

class SymmetryMR(MR):
    def __init__(self):
        super().__init__()
    def expected_output(self, output):
        return output

class AddBlur(SymmetryMR):
    def __init__(self, ksize=(1,5)):
        super().__init__()
        self.ksize = ksize
    def transform_input(self, input):
        return cv2.blur(src=input, ksize=self.ksize)

class AddBrightness(SymmetryMR):
    def __init__(self, delta=-0.3):
        super().__init__()
        self.delta = delta
    def transform_input(self, input):
        modified = input.astype(np.float32) / 255.0
        return np.clip(((modified + self.delta) * 255.0).astype(np.float32), 0.0, 255.0)

class AddContrast(SymmetryMR):
    def __init__(self, saturation=50.0):
        super().__init__()
        self.saturation = saturation
    def transform_input(self, input):
        modified = cv2.cvtColor(src=input, code=cv2.COLOR_RGB2HSV)
        modified[:,:,1] = self.saturation
        cv2.cvtColor(src=modified, dst=modified, code=cv2.COLOR_HSV2RGB)
        return modified

class AddNoise(SymmetryMR):
    def __init__(self, rate=.5, max_delta=.2):
        super().__init__()
        self.rate = rate
        self.max_delta = max_delta
    def transform_input(self, input):
        modified = input.astype(np.float32) / 255.0
        delta = np.random.uniform(size=modified.shape, high=self.max_delta) * np.sign(np.random.uniform(size=modified.shape) - .5)
        return np.clip(((modified + delta) * 255.0).astype(np.float32), 0.0, 255.0)

class HorizontalFlip(MR):
    def __init__(self):
        super().__init__()
    def transform_input(self, input):
        return cv2.flip(src=input, flipCode=1)
    def expected_output(self, output):
        return -output

def init_mr(name, **kwargs):
    return {
        mr.__name__: mr for mr in (AddBlur, AddBrightness, AddContrast, AddNoise, HorizontalFlip)
    }[name](**kwargs)
