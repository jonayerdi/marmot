import struct
from os import listdir
from os.path import join, split

from utils.dataset import Dataset
from utils.statistics import true_positive_rate, false_positive_rate, precision, f1

OUT_FILE = join('results_threshold.csv')

DATA_DIR = join('data', 'dataset_circuit_1')
VERDICTS_DIR = join('data', 'verdicts_circuit_1')
THRESHOLDS_DIR = join('data', 'thresholds_circuit_1')

DATA_DIRS = set(listdir(DATA_DIR))

THRESHOLDS_TOLERANCE = 1.1

IS_DATASET_GOOD = lambda d: d in set((f'data_good.group{i}' for i in range(1, 31)))
IS_DATASET_MUTANT = lambda d: d.startswith('data_m_')
IS_DATASET_ANOMALY = lambda d: d.startswith('data_a_')

def get_threshold_from_oracle_id(oracle_id, thresholds_tolerance=THRESHOLDS_TOLERANCE, thresholds_dir=THRESHOLDS_DIR):
    threshold_file = f'{oracle_id}.threshold.txt'
    threshold_path = join(thresholds_dir, threshold_file)
    with open(threshold_path, mode='r') as fp:
        return float(next(iter(fp)).strip()) * thresholds_tolerance
    
def process_verdicts_before_oob(data_dir, verdicts, data_dirs_root=DATA_DIR, use_data_dirs=DATA_DIRS, reaction_frames=None):
    if data_dir in use_data_dirs:
        dataset = Dataset(data_dir=join(data_dirs_root, data_dir))
        oob = dataset.events.find_first(lambda e: e[0] in ['oob', 'collision'])[1]
        if oob is None:
            return verdicts
        for index, image in enumerate(dataset.images):
            if split(image)[1] == oob:
                if reaction_frames is not None:
                    index -= reaction_frames
                    detection_begin = dataset.events.find_first(lambda e: e[0] in ['anomaly'])[1]
                    if detection_begin is None:
                        detection_begin = 0
                    else:
                        detection_begin = next(filter(
                            lambda img: split(img[1])[1] == detection_begin, 
                            enumerate(dataset.images),
                        ))[0]
                    if index < 0 or index < detection_begin:
                        return []
                return verdicts[:index]
        raise Exception(f'OOB frame {oob} not found in dataset {data_dir}')
    return None
    
def read_verdicts(path, process_verdicts=process_verdicts_before_oob):
    raw = None
    with open(path, mode='rb') as fd:
        raw = fd.read()
    index = 0
    data_dir_end = raw.find(b'\n', index)
    while data_dir_end != -1:
        data_dir = raw[index:data_dir_end].decode(encoding='utf-8')
        index = data_dir_end + 1
        verdicts_len = struct.unpack_from('<L', raw, index)[0]
        #verdicts_len = struct.unpack('<L', raw[index:index+4])[0]
        index += 4
        #verdicts = [
        #    struct.unpack('<f', raw[index+(4*i):index+(4*i)+4])[0]
        #    for i in range(verdicts_len)
        #]
        verdicts = struct.unpack_from(f'<{verdicts_len}f', raw, index)
        index += 4*verdicts_len
        verdicts = process_verdicts(data_dir, verdicts)
        if verdicts is not None:
            yield data_dir, verdicts
        data_dir_end = raw.find(b'\n', index)

def get_results(verdicts_iter, threshold):
    for data_dir, verdicts in verdicts_iter:
        result = False
        for verdict in verdicts:
            if verdict > threshold:
                result = True
                break
        yield data_dir, result

def get_classification_counts(results, is_dataset_good=IS_DATASET_GOOD, is_dataset_anomaly=IS_DATASET_ANOMALY, is_dataset_mutant=IS_DATASET_MUTANT):
    g_fp = g_tn = a_tp = a_fn = m_tp = m_fn = 0
    for data_dir, result in results:
        if is_dataset_anomaly(data_dir):
            if result:
                a_tp += 1
            else:
                a_fn += 1
        elif is_dataset_mutant(data_dir):
            if result:
                m_tp += 1
            else:
                m_fn += 1
        elif is_dataset_good(data_dir):
            if result:
                g_fp += 1
            else:
                g_tn += 1
        else:
            raise Exception(f'Unclassified dataset: "{data_dir}"')
    return g_fp, g_tn, a_tp, a_fn, m_tp, m_fn

def eval_verdicts_threshold():
    with open(OUT_FILE, mode='w', encoding='utf-8', newline='\n') as fd:
        fd.write('ORACLE,NOMINAL_FP,NOMINAL_TN,NOMINAL_FPR,ANOMALY_TP,ANOMALY_FN,ANOMALY_TPR,ANOMALY_PRECISION,ANOMALY_F1,MUTANT_TP,MUTANT_FN,MUTANT_TPR,MUTANT_PRECISION,MUTANT_F1\n')
        fd.flush()
        for filename in listdir(VERDICTS_DIR):
            if filename.endswith('.verdicts.bin'):
                oracle_id = filename[:-len('.verdicts.bin')]
                path = join(VERDICTS_DIR, filename)
                verdicts = read_verdicts(path)
                threshold = get_threshold_from_oracle_id(oracle_id)
                results = get_results(verdicts_iter=verdicts, threshold=threshold)
                g_fp, g_tn, a_tp, a_fn, m_tp, m_fn = get_classification_counts(results)
                # Nominal conditions
                g_fpr = false_positive_rate(false_positives=g_fp, true_negatives=g_tn)
                # Anomalies
                a_tpr = true_positive_rate(true_positives=a_tp, false_negatives=a_fn)
                a_prec = precision(true_positives=a_tp, false_positives=g_fp)
                a_f1 = f1(precision=a_prec, recall=a_tpr)
                # Mutants
                m_tpr = true_positive_rate(true_positives=m_tp, false_negatives=m_fn)
                m_prec = precision(true_positives=m_tp, false_positives=g_fp)
                m_f1 = f1(precision=a_prec, recall=m_tpr)
                # Write results
                fd.write(f'{oracle_id},{g_fp},{g_tn},{g_fpr},{a_tp},{a_fn},{a_tpr},{a_prec},{a_f1},{m_tp},{m_fn},{m_tpr},{m_prec},{m_f1}\n')
                fd.flush()

if __name__ == '__main__':
    eval_verdicts_threshold()
