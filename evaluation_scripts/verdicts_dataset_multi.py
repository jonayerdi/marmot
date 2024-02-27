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

DATA_DIR = join('data', 'dataset_circuit2')

OUT_DIR = join('data', 'verdicts_circuit2')

MUTANT_MODEL_PATH = lambda mutant, index: join('data', 'ads_models', f'tflite_model_mutant_{mutant}_{index}.tflite')
ORIGINAL_MODEL_PATH = join('data', 'ads_models', 'tflite_model_withOurData_ruido.tflite')

MUTANT_MODEL_PATH_STOCHASTIC = lambda mutant, index: join('data', 'mcdropout_leorover', f'tflite_model_mutant_{mutant}_{index}.tflite')
ORIGINAL_MODEL_PATH_STOCHASTIC = MUTANT_MODEL_PATH_STOCHASTIC('ORIG_0', 0)

SO_MODELS_ALL_GLOB = 'data/selforacle_models/*E_ALL_*.tflite'
SO_MODEL_TYPE = lambda name: split(name)[1][:3]

#REGEX_MUTANT_DATA = re.compile(r'^([(HLR)|(HNE)|(TAN)].+)$')
REGEX_MUTANT_DATA = re.compile(r'^data_m_(.+)$')
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
            return mutant_model_path('ORIG_0', index)
        else:
            return original_model_path

SUT_MODELS = {}
def SUT_MODEL(path):
    if path not in SUT_MODELS:
        SUT_MODELS[path] = load_sut_model(path)
    return SUT_MODELS[path]

DATASET_NAMES = [
    'AnomaliesPass',
    'EnvironmentalAnomaliesFail',
    'EnvironmentalAnomaliesPass',
    'MutantsPass',
    'NominalAnomaliesFailMutantsFail',
]
DATASETS = {
    name: sorted((Dataset(data_dir=join(DATA_DIR, name, d)) for d in filter(lambda d: True, listdir(join(DATA_DIR, name)))), key=lambda d: d.data_dir)
    for name in DATASET_NAMES
}

FILTER = SimpleARFilter()

ORACLES_MCDROPOUT = [
    MCDropoutOracle(sut_model_path=ORIGINAL_MODEL_PATH_STOCHASTIC, num_samples=16),
]
ORACLES_ENSEMBLE = [
    EnsembleOracle(sut_model_paths=[
        MUTANT_MODEL_PATH('ORIG_0', index)
        for index in range(10)
    ]),
]
ORACLES_MR = [
    MROracle(mr=mr, sut_model_path=ORIGINAL_MODEL_PATH, img_processing='leorover')
    for mr in [AddBlur(), AddBrightness(), AddContrast(), AddNoise(), HorizontalFlip()]
]
ORACLES_SO_ALL = [
    SelfOracle(model_path=model_path, model_type=SO_MODEL_TYPE(model_path), img_processing='selforacle') 
    for model_path in glob(SO_MODELS_ALL_GLOB)
]

ORACLES = [
    *ORACLES_MR,
    *ORACLES_ENSEMBLE,
    *ORACLES_SO_ALL,
    *ORACLES_MCDROPOUT,
]

def oracle_out_file(oracle):
    clazz = oracle.__class__
    name = clazz.__name__
    return {
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
                    ))
                    fp.write(struct.pack('<L', len(verdicts)))
                    for verdict in verdicts:
                        fp.write(struct.pack('<f', verdict))
                    oracle.reset()
                    FILTER.reset()
