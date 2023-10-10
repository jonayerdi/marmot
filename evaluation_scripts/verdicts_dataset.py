from argparse import ArgumentParser

def print_progress_every(x):
    class PPE:
        def __init__(self, x) -> None:
            self.x = x
            self.checkpoint = x
        def print_progress(self, progress):
            progress *= 100
            update = False
            while progress >= self.checkpoint:
                self.checkpoint += self.x
                update = True
            if update:
                print(f'{self.checkpoint - x}%')
    return PPE(x).print_progress

def verdicts_dataset(oracle, filter, dataset, progress=lambda _: None):
    for index, image in enumerate(dataset.iter_images()):
        name, image = image
        verdict = oracle.next(image).verdict()
        verdict_filtered = filter.next(verdict).compute()
        yield verdict_filtered
        progress(index / len(dataset.images))

def main(args):
    from utils.dataset import Dataset
    from utils.filter import init_filter
    from utils.oracle import init_oracle
    parser = ArgumentParser(description='Misbehaviour detection model training')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, required=True)
    parser.add_argument('-o', help='oracle', dest='oracle', type=str, required=True)
    parser.add_argument('-f', help='filter', dest='filter', type=str, default='SimpleARFilter', required=False)
    parser.add_argument('-pp', help='print progress', dest='progress', type=int, default=0, required=False)
    params = parser.parse_args(args=args)
    dataset = Dataset(data_dir=params.data_dir)
    verdicts = verdicts_dataset(
        oracle=init_oracle(args=params.oracle.split(',')),
        filter=init_filter(name=params.filter),
        dataset=dataset,
        progress=print_progress_every(params.progress)
    )
    for verdict in verdicts:
        print(verdict)

if __name__ == '__main__':
    import sys
    main(args=sys.argv[1:])
