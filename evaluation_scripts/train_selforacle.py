from argparse import ArgumentParser

import numpy as np

from selforacle.anomaly_detector_builder import init_anomaly_detector
from utils.images import preprocess_random_augmentation

def train_selforacle(data_dir, model_name, model_type, epochs=2, batch_size=32, img_processing='selforacle', random_seed=777, augmentation_rate=0.6, sequence_length=30):
    np.random.seed(random_seed)
    if augmentation_rate:
        img_processing = preprocess_random_augmentation(processing=img_processing, augmentation_rate=augmentation_rate)
    anomaly_detector = init_anomaly_detector(model_type=model_type, name=model_name)
    batch_generator = anomaly_detector.get_batch_generator(data_dir=data_dir, batch_size=batch_size, sequence_length=sequence_length, img_processing=img_processing)
    anomaly_detector.train(batch_generator=batch_generator, epochs=epochs)
    anomaly_detector.save_model(format='h5')
    anomaly_detector.save_model(format='tflite')
    #losses = anomaly_detector.calculate_losses(batch_generator=batch_generator)
    #anomaly_detector.save_thresholds(losses=losses)

def main(args):
    parser = ArgumentParser(description='SelfOracle model training')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, required=True)
    parser.add_argument('-n', help='model name', dest='model_name', type=str, required=True)
    parser.add_argument('-t', help='model type', dest='model_type', type=str, required=True)
    parser.add_argument('-e', help='number of epochs', dest='epochs', type=int, default=2)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=32)
    parser.add_argument('-i', help='image processing', dest='img_processing', type=str, default='selforacle')
    parser.add_argument('-r', help='random seed', dest='random_seed', type=int, default=0)
    parser.add_argument('-a', help='augmentation rate', dest='augmentation_rate', type=float, default=0.6)
    parser.add_argument('-s', help='sequence length', dest='sequence_length', type=int, default=30)
    params = parser.parse_args(args=args)
    train_selforacle(
        data_dir=params.data_dir, 
        model_name=params.model_name, model_type=params.model_type,
        epochs=params.epochs, batch_size=params.batch_size, img_processing=params.img_processing,
        random_seed=params.random_seed, augmentation_rate=params.augmentation_rate,
        sequence_length=params.sequence_length,
    )

if __name__ == '__main__':
    import sys
    main(args=sys.argv[1:])
