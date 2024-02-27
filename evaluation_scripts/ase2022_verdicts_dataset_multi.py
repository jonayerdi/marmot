import re
from glob import glob
from os import listdir, makedirs
from os.path import join, split, isfile
import struct

from verdicts_dataset import verdicts_dataset, print_progress_every
from metamorphic.mr import *
from selforacle.anomaly_detector_builder import * 
from utils.dataset import DatasetASE2022
from utils.filter import SimpleARFilter
from utils.oracle import *
from utils.sut import load_sut_model

OUT_DIR = join('data', 'ASE2022', 'verdicts_ase2022')

DATA_DIR = join('data', 'ASE2022', 'benchmark-ASE2022')

DATA_DIR_GOOD = join('data', 'ASE2022', 'datasets-training-sdc', 'datasets', 'dataset5', 'track1')
DATA_DIR_MUTANT = join(DATA_DIR, 'mutants')
DATA_DIR_ANOMALY = DATA_DIR

ORIGINAL_MODEL_PATH = lambda index: join('data', 'ASE2022', 'marmot-dave2-models', f'track1-dave2-mc-053-final-all_{index}.tflite')
MUTANT_MODEL_PATH = lambda mutant, index: join('data', 'ASE2022', 'marmot-dave2-models-mutants', f'{mutant}_{index}.tflite')
#MUTANT_MODEL_PATH = lambda mutant, index: join('data', 'ASE2022', 'marmot-dave2-models-mutants-h5', f'{mutant}_{index}.h5')

ORIGINAL_MODEL_PATH_STOCHASTIC = lambda index: join('data', 'ASE2022', 'marmot-dave2-models-stochastic', f'track1-dave2-mc-053-mc-final-stochastic_{index}.tflite')
MUTANT_MODEL_PATH_STOCHASTIC = lambda mutant, index: join('data', 'ASE2022', 'marmot-dave2-models-mutants-stochastic', f'{mutant}_{index}.tflite')
#MUTANT_MODEL_PATH_STOCHASTIC = lambda mutant, index: join('data', 'ASE2022', 'marmot-dave2-models-mutants-stochastic-h5', f'{mutant}_{index}.h5')

SO_MODELS_GLOB = 'data/ASE2022/marmot-selforacle-models/*.tflite'
SO_MODEL_TYPE = lambda name: split(name)[1][:3]

REGEX_MUTANT_DATA = re.compile(r'(udacity_.+)$')
def SUT_MODEL_PATH(data_dir, index=None, stochastic=False):
    mutant_model_path = MUTANT_MODEL_PATH_STOCHASTIC if stochastic else MUTANT_MODEL_PATH
    original_model_path = ORIGINAL_MODEL_PATH_STOCHASTIC if stochastic else ORIGINAL_MODEL_PATH
    match = REGEX_MUTANT_DATA.match(data_dir)
    if match:
        mutant = match.group(1)
        if index is None:
            index = mutant.split('_')[-1]
        mutant = '_'.join(mutant.split('_')[:-1])
        return mutant_model_path(mutant, index)
    else:
        if index is not None:
            return original_model_path(index)
        else:
            return original_model_path(0)

SUT_MODELS = {}
def SUT_MODEL(path):
    if path not in SUT_MODELS:
        SUT_MODELS[path] = load_sut_model(path)
    return SUT_MODELS[path]

IMG_PROCESSING = 'dave2'
#IMG_PROCESSING = 'dave2large'

DATASET_GOOD = (DatasetASE2022(data_dir=join(DATA_DIR_GOOD, d)) for d in filter(lambda d: d == 'normal', listdir(DATA_DIR_GOOD)))
DATASET_MUTANT = (DatasetASE2022(data_dir=join(DATA_DIR_MUTANT, d)) for d in filter(lambda d: d.startswith('udacity_'), listdir(join(DATA_DIR_MUTANT))))
DATASET_ANOMALY = (DatasetASE2022(data_dir=join(DATA_DIR_ANOMALY, d)) for d in filter(lambda d: d.startswith('xai-track1-'), listdir(DATA_DIR_ANOMALY)))

DATASETS = {
    'normal': sorted(DATASET_GOOD, key=lambda d: d.data_dir),
    'mutant': sorted(DATASET_MUTANT, key=lambda d: d.data_dir),
    'anomaly': sorted(DATASET_ANOMALY, key=lambda d: d.data_dir),
}

FILTER = SimpleARFilter(coefficients=[1])

ORACLES_MCDROPOUT = [
    MCDropoutOracle(
        sut_model_path=ORIGINAL_MODEL_PATH_STOCHASTIC(0),
        num_samples=16,
        img_processing=IMG_PROCESSING,
    ),
]
ORACLES_ENSEMBLE = [
    EnsembleOracle(sut_model_paths=[
        ORIGINAL_MODEL_PATH(index)
        for index in range(10)
    ], img_processing=IMG_PROCESSING),
]
ORACLES_MR = [
    MROracle(mr=mr, sut_model_path=ORIGINAL_MODEL_PATH(0), img_processing=IMG_PROCESSING)
    for mr in [AddBlur(), AddBrightness(), AddContrast(), AddNoise(), HorizontalFlip()]
]
ORACLES_SO = [
    SelfOracle(model_path=model_path, model_type=SO_MODEL_TYPE(model_path), img_processing='selforacle_dave2') 
    for model_path in glob(SO_MODELS_GLOB)
]

ORACLES = [
    *ORACLES_MR,
    *ORACLES_ENSEMBLE,
    *ORACLES_SO,
    *ORACLES_MCDROPOUT,
]

def oracle_out_file(oracle):
    clazz = oracle.__class__
    name = clazz.__name__
    return {
        UWizMCDropoutOracle: lambda: f'{name}.verdicts.bin',
        MCDropoutOracle: lambda: f'{name}.verdicts.bin',
        EnsembleOracle: lambda: f'{name}.verdicts.bin',
        MROracle: lambda: f'{name}.{oracle.mr}.verdicts.bin',
        SelfOracle: lambda: f'{name}.{oracle}.verdicts.bin',
    }[clazz]()

for oracle in ORACLES:
    for dataset_group in DATASETS:
        makedirs(join(OUT_DIR, dataset_group), exist_ok=True)
        OUT_FILE = join(OUT_DIR, dataset_group, oracle_out_file(oracle))
        if not isfile(OUT_FILE):
            with open(OUT_FILE, mode='wb') as fp:
                for dataset in DATASETS[dataset_group]:
                    data_dir = split(dataset.data_dir)[1]
                    print(data_dir)
                    fp.write(data_dir.encode('utf-8'))
                    fp.write('\n'.encode('utf-8'))
                    if type(oracle) == MROracle:
                        oracle.sut_model = SUT_MODEL(SUT_MODEL_PATH(data_dir))
                    elif type(oracle) == EnsembleOracle:
                        oracle.sut_models = [
                            SUT_MODEL(SUT_MODEL_PATH(data_dir, index))
                            for index in range(10)
                        ]
                    elif type(oracle) == MCDropoutOracle:
                        oracle.sut_model = SUT_MODEL(SUT_MODEL_PATH(data_dir, index=0, stochastic=True))
                    verdicts = list(verdicts_dataset(
                        oracle=oracle,
                        filter=FILTER,
                        dataset=dataset,
                        progress=print_progress_every(1),
                    ))
                    fp.write(struct.pack('<L', len(verdicts)))
                    for verdict in verdicts:
                        fp.write(struct.pack('<f', verdict))
                    oracle.reset()
                    FILTER.reset()
