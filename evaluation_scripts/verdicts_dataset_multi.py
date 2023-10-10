import re
from glob import glob
from os import listdir, makedirs
from os.path import join, split, isfile
import struct

from verdicts_dataset import verdicts_dataset
from metamorphic.mr import *
from selforacle.anomaly_detector_builder import * 
from utils.dataset import Dataset
from utils.filter import SimpleARFilter
from utils.oracle import *
from utils.sut import load_sut_model

OUT_DIR = join('data', 'verdicts_circuit_1')

DATA_DIR = join('data', 'dataset_circuit_1')
THRESHOLDS_DIR = join('data', 'thresholds_circuit_1')

ORIGINAL_MODEL_PATH = join('data', 'models', 'tflite_model_mutant_ORIG_0_0.tflite')
MUTANT_MODEL_PATH = lambda m: join('data', 'models', 'mutants', f'tflite_model_mutant_{m}.tflite')

SO_MODELS_GLOB = 'models_SO_training_data/*E_ALL_*.tflite'
SO_MODEL_TYPE = lambda name: split(name)[1][:3]

REGEX_MUTANT_DATA = re.compile(r'^data_m_(.+)$')
def SUT_MODEL_PATH(data_dir, index=None):
    match = REGEX_MUTANT_DATA.match(data_dir)
    if match:
        mutant = match.group(1)
        if mutant.startswith('good_'):
            mutant = mutant[len('good_'):]
        if index is not None:
            mutant = '_'.join(mutant.split('_')[:-1]) + f'_{index}'
        return MUTANT_MODEL_PATH(mutant)
    else:
        if index is not None:
            return MUTANT_MODEL_PATH(f'ORIG_0_{index}')
        else:
            return ORIGINAL_MODEL_PATH

SUT_MODELS = {}
def SUT_MODEL(path):
    if path not in SUT_MODELS:
        SUT_MODELS[path] = load_sut_model(path)
    return SUT_MODELS[path]

IS_DATASET_GOOD = lambda d: d in set((f'data_good.group{i}' for i in range(1, 31)))
IS_DATASET_MUTANT = lambda d: d.startswith('data_m_')
IS_DATASET_ANOMALY = lambda d: d.startswith('data_a_')

DATASETS = sorted(map(
    lambda d: Dataset(data_dir=join(DATA_DIR, d)),
    filter(
        lambda d: IS_DATASET_ANOMALY(d) or IS_DATASET_MUTANT(d) or IS_DATASET_GOOD(d),
        #lambda d: IS_DATASET_MUTANT(d),
        #lambda d: 'good' in d,
        #lambda d: d == 'data_good' or d.startswith('data_a_') or d.startswith('data_m_'),
        listdir(DATA_DIR)
    )
), key=lambda d: d.data_dir)

FILTER = SimpleARFilter()

ORACLES_ENSEMBLE = [
    EnsembleOracle(sut_model_paths=[
        f'data/models/mutants/tflite_model_mutant_ORIG_0_{index}.tflite'
        for index in range(10)
    ]),
]
ORACLES_MR = [
    MROracle(mr=mr, sut_model_path=ORIGINAL_MODEL_PATH, img_processing='leorover')
    for mr in [AddBlur(), AddBrightness(), AddContrast(), AddNoise(), HorizontalFlip()]
]
ORACLES_SO_ALL = [
    SelfOracle(model_path=model_path, model_type=SO_MODEL_TYPE(model_path), img_processing='selforacle') 
    for model_path in glob(SO_MODELS_GLOB)
]

ORACLES = [
    *ORACLES_MR,
    *ORACLES_ENSEMBLE,
    *ORACLES_SO_ALL,
]

def oracle_out_file(oracle):
    clazz = oracle.__class__
    name = clazz.__name__
    return {
        EnsembleOracle: lambda: f'{name}.verdicts.bin',
        MROracle: lambda: f'{name}.{oracle.mr}.verdicts.bin',
        SelfOracle: lambda: f'{name}.{oracle}.verdicts.bin',
    }[clazz]()

makedirs(OUT_DIR, exist_ok=True)

for oracle in ORACLES:
    OUT_FILE = join(OUT_DIR, oracle_out_file(oracle))
    if not isfile(OUT_FILE):
        with open(OUT_FILE, mode='wb') as fp:
            for dataset in DATASETS:
                data_dir = split(dataset.data_dir)[1]
                fp.write(data_dir.encode('utf-8'))
                fp.write('\n'.encode('utf-8'))
                if type(oracle) == MROracle:
                    oracle.sut_model = SUT_MODEL(SUT_MODEL_PATH(data_dir))
                elif type(oracle) == EnsembleOracle:
                    oracle.sut_models = [
                        SUT_MODEL(SUT_MODEL_PATH(data_dir, index))
                        for index in range(10)
                    ]
                verdicts = list(verdicts_dataset(
                    oracle=oracle,
                    filter=FILTER,
                    dataset=dataset,
                ))
                fp.write(struct.pack('<L', len(verdicts)))
                for verdict in verdicts:
                    fp.write(struct.pack('<f', verdict))
                oracle.reset()
                FILTER.reset()
