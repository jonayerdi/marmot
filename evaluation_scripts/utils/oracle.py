import abc
from glob import glob

class Oracle(abc.ABC):
    def __init__(self):
        pass
    def reset(self):
        pass
    @abc.abstractmethod
    def process_image(self, img):
        raise Exception('method must be overriden in child class')
    @abc.abstractmethod
    def next(self, img):
        raise Exception('method must be overriden in child class')
    @abc.abstractmethod
    def verdict(self):
        raise Exception('method must be overriden in child class')
    def __str__(self):
        return 'Oracle'
    
class NullOracle(Oracle):
    def __init__(self, verdict_value=0.0):
        self.verdict_value = verdict_value
    def process_image(self, img):
        return img
    def next(self, img):
        return self
    def verdict(self):
        return self.verdict_value
    def __str__(self):
        return 'NullOracle'

class StochasticOracle(Oracle):
    from scipy.stats import tstd
    def __init__(self, accumulator=tstd, img_processing='leorover'):
        self.accumulator = accumulator
        self.img_processing = img_processing
        self.image = None
    @abc.abstractmethod
    def get_predictions(self, img):
        raise Exception('method must be overriden in child class')
    def process_image(self, img):
        from utils.images import process_image
        return process_image(img, processing=self.img_processing, rgb=False)
    def next(self, img):
        self.image = self.process_image(img)
        return self
    def verdict(self):
        x = self.get_predictions(self.image)
        y = self.accumulator(x)
        return y
    def __str__(self):
        return self.__class__.__name__
    
class EnsembleOracle(StochasticOracle):
    def __init__(self, sut_model_paths, **kwargs):
        from utils.sut import load_sut_model
        super().__init__(**kwargs)
        self.sut_models = [load_sut_model(sut) for sut in sut_model_paths]
    def get_predictions(self, img):
        from utils.sut import execute_sut_model
        return [execute_sut_model(sut, img) for sut in self.sut_models]
    
class SelfOracle(Oracle):
    def __init__(self, model_path, model_type, img_processing='selforacle', **kwargs):
        from os.path import split, splitext
        from selforacle.anomaly_detector_builder import init_anomaly_detector
        super().__init__()
        name = splitext(split(model_path)[1])[0]
        self.anomaly_detector = init_anomaly_detector(name=name, model_type=model_type, init_model=False, **kwargs)
        self.anomaly_detector.load_model(path=model_path, format='tflite')
        self.img_processing = img_processing
        self.image = None
    def process_image(self, img):
        from utils.images import process_image
        image_unprocessed = process_image(img, processing=self.img_processing, rgb=False)
        return self.anomaly_detector.normalize_and_reshape(image_unprocessed)
    def next(self, img):
        self.image = self.process_image(img)
        return self
    def verdict(self):
        from utils.thresholds import calculate_losses_predictions
        prediction = self.anomaly_detector.predict(x=(self.image,), mode='tflite')
        return calculate_losses_predictions(labels=(self.image,), predictions=(prediction,))[0]
    def __str__(self):
        return self.anomaly_detector.name


class MROracle(Oracle):
    def __init__(self, mr, sut_model_path, img_processing='leorover'):
        from utils.sut import load_sut_model
        super().__init__()
        self.mr = mr
        self.sut_model = load_sut_model(sut_model_path)
        self.img_processing = img_processing
        self.image = None, None
    def process_image(self, img):
        from utils.images import process_image
        image_rgb = process_image(img, processing='rgb')
        source_image = process_image(img, processing=self.img_processing)
        followup_image = process_image(
            process_image(self.mr.transform_input(image_rgb), processing='bgr'),
            processing=self.img_processing
        )
        return source_image, followup_image
    def next(self, img):
        self.image = self.process_image(img)
        return self
    def verdict(self):
        from utils.sut import execute_sut_model
        from utils.thresholds import calculate_losses_predictions
        source_output = execute_sut_model(self.sut_model, self.image[0])
        followup_output = execute_sut_model(self.sut_model, self.image[1])
        expected_output = self.mr.expected_output(source_output)
        return calculate_losses_predictions(labels=(expected_output,), predictions=(followup_output,))[0]
    def __str__(self):
        return str(self.mr)
    @staticmethod
    def init_mr(args):
        from metamorphic.mr import init_mr
        return init_mr(name=args[0], **{ a[0]: eval(a[1]) for a in map(lambda a: a.split('='), args[1:])})

def init_oracle(args):
    return {
        'NullOracle': lambda: NullOracle(verdict_value=0.0),
        'EnsembleOracle': lambda: EnsembleOracle(sut_model_paths=glob(args[1])),
        'SelfOracle': lambda: SelfOracle(model_path=args[1], model_type=args[2], img_processing=args[3]),
        'MROracle': lambda: MROracle(mr=MROracle.init_mr(args[1:-2]), sut_model_path=args[-2], img_processing=args[-1]),
    }[args[0]]()
