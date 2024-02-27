from os import listdir
from os.path import join

from ase2022_eval_verdicts_threshold import read_verdicts, process_verdicts_before_oob, get_results, get_classification_counts
from utils.statistics import true_positive_rate, precision, auc_prc

#EXPERIMENT = 'normal'
EXPERIMENT = 'anomaly'
#EXPERIMENT = 'mutant'

OUT_FILE = join(f'results_auc_reaction_{EXPERIMENT}.csv')

DATA_DIR = join('data', 'ASE2022', 'benchmark-ASE2022')
DATA_DIR_GOOD = join('data', 'ASE2022', 'datasets-training-sdc', 'datasets', 'dataset5', 'track1')
DATA_DIR_MUTANT = join(DATA_DIR, 'mutants')
DATA_DIR_ANOMALY = DATA_DIR

DATA_DIRS_PASS = DATA_DIR_GOOD
VERDICTS_DIR_PASS = join('data', f'verdicts_ase2022', 'normal')
IS_DATASET_PASS = lambda d: d == 'normal'
DATASET_PASS_USE_DATA = lambda dataset, data: data[3309:] # ~80% of the 16542 images from the dataset
DATASET_PASS_PARTS = 10 # 16542 - 3309 = 13233 images for evaluation. Split into 10 recordings of ~1323 images each

DATA_DIRS_FAIL = {
    'normal': DATA_DIR_GOOD,
    'anomaly': DATA_DIR_ANOMALY,
    'mutant': DATA_DIR_MUTANT,
}[EXPERIMENT]
VERDICTS_DIR_FAIL = join('data', f'verdicts_ase2022', f'{EXPERIMENT}')
IS_DATASET_FAIL = {
    'normal': lambda d: d == 'normal',
    'anomaly': lambda d: d.startswith('xai-track1-'),
    'mutant': lambda d: d.startswith('udacity_'),
}[EXPERIMENT]
DATASET_FAIL_USE_DATA = {
    'normal': lambda dataset, data: data[3309:],
    'anomaly': lambda dataset, data: data,
    'mutant': lambda dataset, data: data,
}[EXPERIMENT]

THRESHOLD_SAMPLES = 500
#REACTION_FRAMES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
REACTION_FRAMES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

def threshold_all_verdicts(verdicts_iter):
    threshold_min = float("inf")
    threshold_max = float("-inf")
    for data_dir, scenarios in verdicts_iter:
        for verdicts in scenarios:
            if len(verdicts) > 0:
                verdict_max = max(verdicts)
                threshold_min = min(threshold_min, verdict_max)
                threshold_max = max(threshold_max, verdict_max)
    return threshold_min, threshold_max

def sample_thresholds(threshold_min, threshold_max, count=THRESHOLD_SAMPLES):
    diff = threshold_max - threshold_min
    step = diff / (count - 2)
    for i in range(count):
        yield threshold_min + (i * step)

def eval_verdicts_auc_reaction():
    with open(OUT_FILE, mode='w', encoding='utf-8', newline='\n') as fd:
        fd.write('ORACLE,REACTION,AUC_PRC\n')
        fd.flush()
        for filename in listdir(VERDICTS_DIR_PASS):
            if filename.endswith('.verdicts.bin'):
                oracle_id = filename[:-len('.verdicts.bin')]
                path_pass = join(VERDICTS_DIR_PASS, filename)
                path_fail = join(VERDICTS_DIR_FAIL, filename)
                for reaction_frames in REACTION_FRAMES:
                    verdicts_pass = list(read_verdicts(
                        path=path_pass, 
                        process_verdicts=lambda data_dir, verdicts: list(process_verdicts_before_oob(
                            data_dir=data_dir, verdicts=verdicts,
                            data_dirs=DATA_DIRS_PASS, use_data=DATASET_PASS_USE_DATA,
                            parts=DATASET_PASS_PARTS, reaction_frames=reaction_frames,
                        )),
                    ))
                    verdicts_fail = list(read_verdicts(
                        path=path_fail, 
                        process_verdicts=lambda data_dir, verdicts: list(process_verdicts_before_oob(
                            data_dir=data_dir, verdicts=verdicts,
                            data_dirs=DATA_DIRS_FAIL, use_data=DATASET_FAIL_USE_DATA,
                            reaction_frames=reaction_frames,
                        )),
                    ))
                    #verdicts = list(filter(lambda v: len(v[1]) > 0, verdicts))
                    threshold_min_pass, threshold_max_pass = threshold_all_verdicts(verdicts_pass)
                    threshold_min_fail, threshold_max_fail = threshold_all_verdicts(verdicts_fail)
                    sampled_thresholds = sample_thresholds(
                        threshold_min=min(threshold_min_pass, threshold_min_fail),
                        threshold_max=max(threshold_max_pass, threshold_max_fail),
                    )
                    a_tprs = []
                    a_precisions = []
                    for threshold in sampled_thresholds:
                        results_pass = get_results(verdicts_iter=verdicts_pass, threshold=threshold)
                        results_fail = get_results(verdicts_iter=verdicts_fail, threshold=threshold)
                        g_fp, g_tn = get_classification_counts(results=results_pass, filter_dataset=IS_DATASET_PASS)
                        a_tp, a_fn = get_classification_counts(results=results_fail, filter_dataset=IS_DATASET_FAIL)
                        # Anomalies
                        a_tprs.append(true_positive_rate(true_positives=a_tp, false_negatives=a_fn))
                        a_precisions.append(precision(true_positives=a_tp, false_positives=g_fp))
                    # Anomalies
                    a_auc_prc = auc_prc(true_positive_rates=a_tprs, precisions=a_precisions)
                    # Write results
                    fd.write(f'{oracle_id},{-reaction_frames},{a_auc_prc}\n')
                    fd.flush()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        EXPERIMENT = sys.argv[1]
        OUT_FILE = join(f'results_auc_reaction_{EXPERIMENT}.csv')
        DATA_DIRS_FAIL = {
            'normal': DATA_DIR_GOOD,
            'anomaly': DATA_DIR_ANOMALY,
            'mutant': DATA_DIR_MUTANT,
        }[EXPERIMENT]
        VERDICTS_DIR_FAIL = join('data', f'verdicts_ase2022', f'{EXPERIMENT}')
        IS_DATASET_FAIL = {
            'normal': lambda d: d == 'normal',
            'anomaly': lambda d: d.startswith('xai-track1-'),
            'mutant': lambda d: d.startswith('udacity_'),
        }[EXPERIMENT]
        DATASET_FAIL_USE_DATA = {
            'normal': lambda dataset, data: data[3309:],
            'anomaly': lambda dataset, data: data,
            'mutant': lambda dataset, data: data,
        }[EXPERIMENT]
    eval_verdicts_auc_reaction()
