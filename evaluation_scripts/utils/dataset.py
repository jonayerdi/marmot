import re
from bisect import bisect_left
from os.path import join, split

from utils.images import get_images, load_image

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def get_events(lines, separator=','):
    if lines:
        for line in lines:
            data = line.strip().split(separator)
            if len(data) == 2:
                yield (data[0], data[1])

def image_index(img):
    # 26120231033-img0.jpg
    # 261202390-img538.jpg
    from os.path import split, splitext
    img = splitext(split(img)[1])[0]
    img = img[img.find('-') + len('-img'):]
    return int(img)

def relative_frame(a, b):
    if a is None or b is None:
        return None
    return image_index(a) - image_index(b)

class Events:
    KEY_NAME = alphanum_key
    KEY_EVENT = lambda e: Events.KEY_NAME(e[1])
    class KeyWrapper:
        def __init__(self, events):
            self.events = events
        def __getitem__(self, i):
            return Events.KEY_EVENT(self.events[i])
        def __len__(self):
            return len(self.events)
    def __init__(self, events):
        self.events = sorted(events, key=Events.KEY_EVENT)
    def __getitem__(self, i):
        return self.events[i]
    def find_all(self, cond):
        return filter(cond, self.events)
    def find_first(self, cond):
        try:
            return next(self.find_all(cond))
        except StopIteration:
            return (None, None)
    def find_index_left(self, name):
        return bisect_left(Events.KeyWrapper(self.events), Events.KEY_NAME(name))
    def find_index_right(self, name):
        return bisect_left(Events.KeyWrapper(self.events), Events.KEY_NAME(name))
    def insert(self, event):
        self.events.insert(self.find_index_left(name=event[1]), event)
    def __str__(self) -> str:
        s = ''
        for event in self.events:
            s += f'{Events.as_str(event)}\n'
        return s
    @staticmethod
    def as_str(event):
        return f'{event[0]},{event[1]}'
    @staticmethod
    def from_file(file):
        if type(file) is str:
            try:
                with open(file, mode='r') as fp:
                    return Events.from_file(fp)
            except FileNotFoundError:
                return Events(events=[])
        return Events(events=get_events(lines=iter(file)))

class Dataset:
    def __init__(self, data_dir, image_files=lambda f: f.endswith('.jpg'), events_file='events.csv') -> None:
        self.data_dir = data_dir
        if events_file is not None:
            self.events = Events.from_file(join(self.data_dir, events_file))
        else:
            self.events = Events(events=[])
        if image_files is not None:
            self.images = sorted(get_images(self.data_dir, images_filter=image_files), key=alphanum_key)
        else:
            self.images = []
    def iter_images(self, processing=None):
        for image in self.images:
            yield split(image)[1], load_image(image_path=image, processing=processing)

'''
Classes for "ThirdEye: Attention Maps for Safe Autonomous Driving Systems" dataset by Stocco et al. (ASE 2022):
https://github.com/tsigalko18/ase22
'''

class EventsASE2022(Events):
    @staticmethod
    def get_events(lines, separator=','):
        if lines:
            headers = next(lines).strip().split(separator)
            h_center = headers.index('center')
            try:
                h_crashed = headers.index('crashed')
            except:
                h_crashed = None
            try:
                h_tot_OBEs = headers.index('tot_OBEs')
            except:
                h_tot_OBEs = None
            try:
                h_tot_crashes = headers.index('tot_crashes')
            except:
                h_tot_crashes = None
            crashed = 0
            tot_OBEs = 0
            tot_crashes = 0
            for line in lines:
                data = line.strip().split(separator)
                if len(data) > 1:
                    image = split(data[h_center])[1]
                    if h_crashed is not None:
                        crashed_new = int(data[h_crashed])
                        if crashed_new != crashed:
                            crashed = crashed_new
                            evt = 'oob' if crashed else 'recover'
                            yield (evt, image)
                    if h_tot_OBEs is not None:
                        obes = int(data[h_tot_OBEs])
                        if obes > tot_OBEs:
                            tot_OBEs = obes
                            yield ('tot_OBEs', image)
                    if h_tot_crashes is not None:
                        crashes = int(data[h_tot_crashes])
                        if crashes > tot_crashes:
                            tot_crashes = crashes
                            yield ('tot_crashes', image)
    @staticmethod
    def from_file(file):
        if type(file) is str:
            try:
                with open(file, mode='r') as fp:
                    return EventsASE2022.from_file(fp)
            except FileNotFoundError:
                return EventsASE2022(events=[])
        return EventsASE2022(events=EventsASE2022.get_events(lines=iter(file)))

class DatasetASE2022(Dataset):
    def __init__(self, data_dir, image_files=lambda f: f.endswith('.jpg'), events_file='driving_log.csv') -> None:
        self.data_dir = data_dir
        if events_file is not None:
            self.events = EventsASE2022.from_file(join(self.data_dir, events_file))
        else:
            self.events = EventsASE2022(events=[])
        if image_files is not None:
            self.images = sorted(get_images(join(self.data_dir, 'IMG'), images_filter=image_files), key=alphanum_key)
        else:
            self.images = []
