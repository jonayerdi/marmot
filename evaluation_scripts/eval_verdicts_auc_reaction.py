from os import listdir
from os.path import join

from eval_verdicts_threshold import read_verdicts, process_verdicts_before_oob, get_results, get_classification_counts
from utils.statistics import true_positive_rate, precision, auc_prc

OUT_FILE = join('results_auc_reaction.csv')

DATA_DIR = join('data', 'dataset_circuit_1')
VERDICTS_DIR = join('data', 'verdicts_circuit_1')

THRESHOLD_SAMPLES = 500

REACTION_FRAMES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

DATA_DIRS = set(listdir(DATA_DIR))

IS_DATASET_GOOD = lambda d: d in set((f'data_good.group{i}' for i in range(1, 31)))
IS_DATASET_MUTANT = lambda d: d.startswith('data_m_')
IS_DATASET_ANOMALY = lambda d: d.startswith('data_a_')

def threshold_all_verdicts(verdicts_iter):
    threshold_min = float("inf")
    threshold_max = float("-inf")
    for data_dir, verdicts in verdicts_iter:
        if len(verdicts) > 0:
            verdict_max = max(verdicts)
            threshold_min = min(threshold_min, verdict_max)
            threshold_max = max(threshold_max, verdict_max)
    return threshold_min, threshold_max

def sample_thresholds(verdicts, count=THRESHOLD_SAMPLES):
    threshold_min, threshold_max = threshold_all_verdicts(verdicts)
    diff = threshold_max - threshold_min
    step = diff / (count - 2)
    for i in range(count):
        yield threshold_min + (i * step)

def eval_verdicts_auc_reaction():
    with open(OUT_FILE, mode='w', encoding='utf-8', newline='\n') as fd:
        fd.write('ORACLE,REACTION,ANOMALY_AUC_PRC,MUTANT_AUC_PRC\n')
        fd.flush()
        for filename in listdir(VERDICTS_DIR):
            if filename.endswith('.verdicts.bin'):
                oracle_id = filename[:-len('.verdicts.bin')]
                path = join(VERDICTS_DIR, filename)
                for reaction_frames in REACTION_FRAMES:
                    verdicts = list(read_verdicts(
                        path=path, 
                        process_verdicts=lambda dd, v: process_verdicts_before_oob(
                            data_dir=dd, verdicts=v,
                            data_dirs_root=DATA_DIR, use_data_dirs=DATA_DIRS,
                            reaction_frames=reaction_frames,
                        )
                    ))
                    #verdicts = list(filter(lambda v: len(v[1]) > 0, verdicts))
                    sampled_thresholds = sample_thresholds(verdicts)
                    a_tprs = []
                    a_precisions = []
                    m_tprs = []
                    m_precisions = []
                    for threshold in sampled_thresholds:
                        results = get_results(verdicts_iter=verdicts, threshold=threshold)
                        g_fp, g_tn, a_tp, a_fn, m_tp, m_fn = get_classification_counts(
                            results=results,
                            is_dataset_good=IS_DATASET_GOOD,
                            is_dataset_anomaly=IS_DATASET_ANOMALY,
                            is_dataset_mutant=IS_DATASET_MUTANT,
                        )
                        # Anomalies
                        a_tprs.append(true_positive_rate(true_positives=a_tp, false_negatives=a_fn))
                        a_precisions.append(precision(true_positives=a_tp, false_positives=g_fp))
                        # Mutants
                        m_tprs.append(true_positive_rate(true_positives=m_tp, false_negatives=m_fn))
                        m_precisions.append(precision(true_positives=m_tp, false_positives=g_fp))
                    # Anomalies
                    a_auc_prc = auc_prc(true_positive_rates=a_tprs, precisions=a_precisions)
                    # Mutants
                    m_auc_prc = auc_prc(true_positive_rates=m_tprs, precisions=m_precisions)
                    # Write results
                    fd.write(f'{oracle_id},{-reaction_frames},{a_auc_prc},{m_auc_prc}\n')
                    fd.flush()

if __name__ == '__main__':
    eval_verdicts_auc_reaction()
