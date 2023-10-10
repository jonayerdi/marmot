from utils.images import process_image
from utils.images_batch_generator import ImagesBatchGenerator

class AutoencoderBatchGenerator(ImagesBatchGenerator):
    def __init__(self, data_dir, anomaly_detector, batch_size: int, img_processing='selforacle'):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size, 
            img_shape=anomaly_detector.get_input_shape(), 
            img_processing=lambda image, *args: anomaly_detector.normalize_and_reshape(
                process_image(image=image, processing=img_processing)
            ),
        )
