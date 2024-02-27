import struct
from os import listdir
from os.path import join, split, exists

from utils.dataset import DatasetASE2022
from utils.filter import SimpleARFilter
from utils.statistics import true_positive_rate, false_positive_rate, precision, f1

#EXPERIMENT = 'normal'
EXPERIMENT = 'anomaly'
#EXPERIMENT = 'mutant'

OUT_FILE = join(f'results_{EXPERIMENT}.csv')

FILTER = SimpleARFilter()

THRESHOLDS_DIR = join('data', 'thresholds_ase2022')
THRESHOLDS_TOLERANCE = 1.1

MINIMUM_WINDOW_BEFORE_OOB = 100 # ~10 seconds at 10 FPS

FIRST_OOB_ONLY = False

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

def get_threshold_from_oracle_id(oracle_id, thresholds_tolerance=THRESHOLDS_TOLERANCE, thresholds_dir=THRESHOLDS_DIR):
    threshold_file = f'{oracle_id}.threshold.txt'
    threshold_path = join(thresholds_dir, threshold_file)
    with open(threshold_path, mode='r') as fp:
        return float(next(iter(fp)).strip()) * thresholds_tolerance
    
def process_verdicts_before_oob(data_dir, verdicts, data_dirs, use_data= lambda dataset, data: data, minimum_window_before_oob=MINIMUM_WINDOW_BEFORE_OOB, reaction_frames=0, parts=1, first_oob_only=FIRST_OOB_ONLY):
    if exists(join(data_dirs, data_dir)):
        dataset = DatasetASE2022(data_dir=join(data_dirs, data_dir))
        oob = set(
            event[1]
            for event in dataset.events.find_all(lambda e: e[0] == 'oob')
        )
        recover = set(
            event[1]
            for event in dataset.events.find_all(lambda e: e[0] == 'recover')
        )
        status = 'recovered'
        index_start = 0
        images = use_data(data_dir, dataset.images)
        verdicts = use_data(data_dir, verdicts)
        if oob:
            # Yield recordings of at least minimum_window_before_oob images, beginning
            # with the vehicle in bounds, until the frame where it goes out-of-bounds
            for index, image in enumerate(images):
                img = split(image)[1]
                if img in oob and status == 'recovered':
                    if index - index_start > minimum_window_before_oob:
                        yield verdicts[index_start:index-reaction_frames]
                        if first_oob_only:
                            return
                    status = 'oob'
                elif img in recover and status == 'oob':
                    status = 'recovered'
                    index_start = index + 2
        else:
            # Split recording into parts
            index = 0
            part_length = int(len(verdicts) / parts)
            for _ in range(parts):
                yield verdicts[index:index+part_length]
                index += part_length
    
def read_verdicts(path, process_verdicts, filter=FILTER):
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
        verdicts = list(struct.unpack_from(f'<{verdicts_len}f', raw, index))
        index += 4*verdicts_len
        for i in range(len(verdicts)):
            verdicts[i] = filter.next(verdicts[i]).compute()
        filter.reset()
        verdicts = process_verdicts(data_dir, verdicts)
        if verdicts is not None:
            yield data_dir, verdicts
        data_dir_end = raw.find(b'\n', index)

def get_results(verdicts_iter, threshold):
    for data_dir, scenarios in verdicts_iter:
        results = []
        for verdicts in scenarios:
            result = False
            for verdict in verdicts:
                if verdict > threshold:
                    result = True
                    break
            results.append(result)
        yield data_dir, results

def get_classification_counts(results, filter_dataset=lambda d: True):
    p = n = 0
    for data_dir, results in results:
        if filter_dataset(data_dir):
            for result in results:
                if result:
                    p += 1
                else:
                    n += 1
    return p, n

def eval_verdicts_threshold():
    with open(OUT_FILE, mode='w', encoding='utf-8', newline='\n') as fd:
        fd.write('ORACLE,FP,TN,FPR,TP,FN,TPR,PRECISION,F1\n')
        fd.flush()
        for filename in listdir(VERDICTS_DIR_PASS):
            if filename.endswith('.verdicts.bin'):
                oracle_id = filename[:-len('.verdicts.bin')]
                path_pass = join(VERDICTS_DIR_PASS, filename)
                path_fail = join(VERDICTS_DIR_FAIL, filename)
                verdicts_pass = read_verdicts(
                    path=path_pass, 
                    process_verdicts=lambda data_dir, verdicts: process_verdicts_before_oob(
                        data_dir=data_dir, verdicts=verdicts,
                        data_dirs=DATA_DIRS_PASS, use_data=DATASET_PASS_USE_DATA,
                        parts=DATASET_PASS_PARTS,
                    ),
                )
                verdicts_fail = read_verdicts(
                    path=path_fail, 
                    process_verdicts=lambda data_dir, verdicts: process_verdicts_before_oob(
                        data_dir=data_dir, verdicts=verdicts,
                        data_dirs=DATA_DIRS_FAIL, use_data=DATASET_FAIL_USE_DATA,
                    ),
                )
                threshold = get_threshold_from_oracle_id(oracle_id)
                results_pass = get_results(verdicts_iter=verdicts_pass, threshold=threshold)
                results_fail = get_results(verdicts_iter=verdicts_fail, threshold=threshold)
                g_fp, g_tn = get_classification_counts(results=results_pass, filter_dataset=IS_DATASET_PASS)
                a_tp, a_fn = get_classification_counts(results=results_fail, filter_dataset=IS_DATASET_FAIL)
                # Nominal conditions
                g_fpr = false_positive_rate(false_positives=g_fp, true_negatives=g_tn)
                # Anomalies
                a_tpr = true_positive_rate(true_positives=a_tp, false_negatives=a_fn)
                a_prec = precision(true_positives=a_tp, false_positives=g_fp)
                a_f1 = f1(precision=a_prec, recall=a_tpr)
                # Write results
                fd.write(f'{oracle_id},{g_fp},{g_tn},{g_fpr},{a_tp},{a_fn},{a_tpr},{a_prec},{a_f1}\n')
                fd.flush()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        EXPERIMENT = sys.argv[1]
        OUT_FILE = join(f'results_{EXPERIMENT}.csv')
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
    eval_verdicts_threshold()
