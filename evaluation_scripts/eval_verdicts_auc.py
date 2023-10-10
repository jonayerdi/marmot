from os import listdir
from os.path import join

from eval_verdicts_threshold import read_verdicts, process_verdicts_before_oob, get_results, get_classification_counts
from utils.statistics import true_positive_rate, false_positive_rate, precision, auc_prc, auc_roc

OUT_FILE = join('results_auc.csv')

DATA_DIR = join('data', 'dataset_circuit_1')
VERDICTS_DIR = join('data', 'verdicts_circuit_1')

THRESHOLD_SAMPLES = 500

DATA_DIRS = set(listdir(DATA_DIR))

IS_DATASET_GOOD = lambda d: d in set((f'data_good.group{i}' for i in range(1, 31)))
IS_DATASET_MUTANT = lambda d: d.startswith('data_m_')
IS_DATASET_ANOMALY = lambda d: d.startswith('data_a_')

def threshold_all_verdicts(verdicts_iter):
    threshold_min = float("inf")
    threshold_max = float("-inf")
    for data_dir, verdicts in verdicts_iter:
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

def eval_verdicts_auc():
    with open(OUT_FILE, mode='w', encoding='utf-8', newline='\n') as fd:
        fd.write('ORACLE,ANOMALY_AUC_PRC,ANOMALY_AUC_ROC,MUTANT_AUC_PRC,MUTANT_AUC_ROC\n')
        fd.flush()
        for filename in listdir(VERDICTS_DIR):
            if filename.endswith('.verdicts.bin'):
                oracle_id = filename[:-len('.verdicts.bin')]
                path = join(VERDICTS_DIR, filename)
                verdicts = list(read_verdicts(
                    path=path, 
                    process_verdicts=lambda dd, v: process_verdicts_before_oob(
                        data_dir=dd, verdicts=v,
                        data_dirs_root=DATA_DIR, use_data_dirs=DATA_DIRS
                    )
                ))
                sampled_thresholds = sample_thresholds(verdicts)
                g_fprs = []
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
                    # Nominal conditions
                    g_fprs.append(false_positive_rate(false_positives=g_fp, true_negatives=g_tn))
                    # Anomalies
                    a_tprs.append(true_positive_rate(true_positives=a_tp, false_negatives=a_fn))
                    a_precisions.append(precision(true_positives=a_tp, false_positives=g_fp))
                    # Mutants
                    m_tprs.append(true_positive_rate(true_positives=m_tp, false_negatives=m_fn))
                    m_precisions.append(precision(true_positives=m_tp, false_positives=g_fp))
                # Anomalies
                a_auc_prc = auc_prc(true_positive_rates=a_tprs, precisions=a_precisions)
                a_auc_roc = auc_roc(false_positive_rates=g_fprs, true_positive_rates=a_tprs)
                # Mutants
                m_auc_prc = auc_prc(true_positive_rates=m_tprs, precisions=m_precisions)
                m_auc_roc = auc_roc(false_positive_rates=g_fprs, true_positive_rates=m_tprs)
                # Write results
                fd.write(f'{oracle_id},{a_auc_prc},{a_auc_roc},{m_auc_prc},{m_auc_roc}\n')
                fd.flush()

if __name__ == '__main__':
    eval_verdicts_auc()
