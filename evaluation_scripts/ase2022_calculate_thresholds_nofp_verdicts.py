from os import makedirs
from os.path import abspath, join, isfile
from typing import Set

from eval_dataset import print_progress_every
from ase2022_eval_verdicts_threshold import read_verdicts

def calculate_thresholds_nofp(oracle: str, verdicts: str, datasets: Set[str], out_prefix: str, use_data=lambda dataset, data: data, progress=lambda _: None):
    threshold_path = f'{out_prefix}.threshold.txt'
    if not isfile(threshold_path):
        max_verdict = 0
        for data_dir, values in read_verdicts(join(verdicts, oracle), lambda d, v: v):
            if data_dir in datasets:
                values = use_data(data_dir, values)
                for verdict in values:
                    max_verdict = max(max_verdict, verdict)
        makedirs(abspath(join(threshold_path, '..')), exist_ok=True)
        with open(threshold_path, mode='w', newline='\n', encoding='utf-8') as fp:
            fp.write(f'{max_verdict}\n')

SO_MODEL_TYPES = ['CAE', 'DAE', 'SAE', 'VAE']
SO_MODEL_INDICES = list(range(10))
SO_MODEL_NAMES = []
for so_type in SO_MODEL_TYPES:
    for so_idx in SO_MODEL_INDICES:
        SO_MODEL_NAMES.append(f'{so_type}_{so_idx}')
VERDICTS = join('data', 'verdicts_ase2022', 'normal')
DATASETS = set(['normal'])
USE_DATA = lambda dataset, data: data[:3309] # ~20% of the 16542 images from the dataset
THRESHOLDS_DIR = join('data', 'thresholds_ase2022')

if True:
    for mr in ['AddBlur', 'AddBrightness', 'AddContrast', 'AddNoise', 'HorizontalFlip']:
        calculate_thresholds_nofp(
            oracle=f'MROracle.{mr}.verdicts.bin',
            out_prefix=f'{THRESHOLDS_DIR}/MROracle.{mr}',
            verdicts=VERDICTS,
            datasets=DATASETS,
            use_data=USE_DATA,
            progress=print_progress_every(10),
        )

if True:
    calculate_thresholds_nofp(
        oracle=f'MCDropoutOracle.verdicts.bin',
        out_prefix=f'{THRESHOLDS_DIR}/MCDropoutOracle',
        verdicts=VERDICTS,
        datasets=DATASETS,
        use_data=USE_DATA,
        progress=print_progress_every(10),
    )

if True:
    calculate_thresholds_nofp(
        oracle=f'EnsembleOracle.verdicts.bin',
        out_prefix=f'{THRESHOLDS_DIR}/EnsembleOracle',
        verdicts=VERDICTS,
        datasets=DATASETS,
        use_data=USE_DATA,
        progress=print_progress_every(10),
    )

if True:
    for so_model_name in SO_MODEL_NAMES:
        calculate_thresholds_nofp(
            oracle=f'SelfOracle.{so_model_name}.verdicts.bin',
            out_prefix=f'{THRESHOLDS_DIR}/SelfOracle.{so_model_name}',
            verdicts=VERDICTS,
            datasets=DATASETS,
            use_data=USE_DATA,
            progress=print_progress_every(10),
        )
