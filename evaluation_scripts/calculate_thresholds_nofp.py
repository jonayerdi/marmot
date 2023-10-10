from glob import glob
from os import makedirs
from os.path import abspath, join, split, splitext, isfile
from typing import List

from verdicts_dataset import print_progress_every
from metamorphic.mr import *
from utils.dataset import Dataset
from utils.filter import Filter, SimpleARFilter
from utils.oracle import *

def calculate_thresholds_nofp(oracle: Oracle, filter: Filter, datasets: List[Dataset], out_prefix: str, progress=lambda _: None):
    threshold_path = f'{out_prefix}.threshold.txt'
    if not isfile(threshold_path):
        max_verdict = 0
        for dataset_index, dataset in enumerate(datasets):
            for image_index, image in enumerate(dataset.iter_images()):
                name, image = image
                verdict = oracle.next(image).verdict()
                if filter:
                    verdict = filter.next(verdict).compute()
                max_verdict = max(max_verdict, verdict)
            progress(dataset_index / len(datasets))
        makedirs(abspath(join(threshold_path, '..')), exist_ok=True)
        with open(threshold_path, mode='w', newline='\n', encoding='utf-8') as fp:
            fp.write(f'{max_verdict}\n')

SO_MODELS_GLOB = 'models_SO_training_data/*.tflite'
SO_MODEL_TYPE = lambda name: split(name)[1][:3]
DATASETS = [Dataset(f'data/dataset_circuit_1/data_good.group{i}', events_file=None) for i in range(31, 38)]
THRESHOLDS_DIR = 'data/thresholds_circuit_1'

if False:
    for mr in [AddBlur(), AddBrightness(), AddContrast(), AddNoise(), HorizontalFlip()]:
        calculate_thresholds_nofp(
            oracle=MROracle(mr=mr, sut_model_path='data/models/tflite_model_mutant_ORIG_0_0.tflite'),
            out_prefix=f'{THRESHOLDS_DIR}/MROracle.{mr.__class__.__name__}',
            filter=SimpleARFilter(),
            datasets=DATASETS,
            progress=print_progress_every(10),
        )

if False:
    calculate_thresholds_nofp(
        oracle=EnsembleOracle(sut_model_paths=[
            f'data/models/mutants/tflite_model_mutant_ORIG_0_{index}.tflite'
            for index in range(10)
        ]),
        out_prefix=f'{THRESHOLDS_DIR}/EnsembleOracle',
        filter=SimpleARFilter(),
        datasets=DATASETS,
        progress=print_progress_every(10),
    )

if False:
    for model_path in glob(SO_MODELS_GLOB):
        model_name = splitext(split(model_path)[1])[0]
        print(model_name)
        model_type = SO_MODEL_TYPE(model_path)
        calculate_thresholds_nofp(
            oracle=SelfOracle(model_path=model_path, model_type=model_type),
            out_prefix=f'{THRESHOLDS_DIR}/SelfOracle.{model_name}',
            filter=SimpleARFilter(),
            datasets=DATASETS,
            progress=print_progress_every(10),
        )
