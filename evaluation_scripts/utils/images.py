from os import listdir
from os.path import join

import cv2
import numpy as np

# Image cropping
#CROP_TOP = 220
CROP_TOP = 200

# SelfOracle image dimensions
SO_IMAGE_HEIGHT, SO_IMAGE_WIDTH, SO_IMAGE_CHANNELS = 80, 160, 3

# Line-follower image dimensions
LF_IMAGE_HEIGHT, LF_IMAGE_WIDTH, LF_IMAGE_CHANNELS = 120, 160, 1
LF_IMAGE_SHAPE = (LF_IMAGE_HEIGHT, LF_IMAGE_WIDTH, LF_IMAGE_CHANNELS)
HSV_MIN, HSV_MAX = (2, 0, 36), (35, 255, 255)

# Dave2 image dimensions
DAVE2_IMAGE_HEIGHT, DAVE2_IMAGE_WIDTH = 80, 160
DAVE2_IMAGE_HEIGHT2, DAVE2_IMAGE_WIDTH2 = 160, 320

def simple_mask(img, hsv_min, hsv_max):
    return cv2.inRange(img, hsv_min, hsv_max,)

def double_range_mask(img, hsv_min, hsv_max):
    lower_mask = cv2.inRange(img, (0, hsv_min[1], hsv_min[2]), hsv_max,)
    upper_mask = cv2.inRange(img, hsv_min, (179, hsv_max[1], hsv_max[2],))
    return lower_mask + upper_mask

def mask_function(img, hsv_min, hsv_max):
    if hsv_min[0] < hsv_max[0]:
        return simple_mask(img, hsv_min, hsv_max)
    else:
        return double_range_mask(img, hsv_min, hsv_max)
    
def preprocess_null(img, rgb=False):
    if rgb:
        return img, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    
def preprocess_bgr(img, rgb=False):
    # Convert color format
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Done
    if rgb:
        return img_bgr, img
    return img_bgr

def preprocess_rgb(img, rgb=False):
    # Convert color format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Done
    if rgb:
        return img, img
    return img

def preprocess_crop(img, rgb=False):
    img = img[CROP_TOP:, :, :]
    img = img.astype(np.float32)
    img = cv2.resize(img, (LF_IMAGE_WIDTH, LF_IMAGE_HEIGHT))
    if rgb:
        return img, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess_augment(img, rgb=False):
    def random_flip(img):
        return cv2.flip(img, 1) if np.random.uniform() < 0.5 else img
    def random_translate(img, range=(50, 5)):
        xdelta = np.random.uniform(low=-range[0], high=range[0])
        ydelta = np.random.uniform(low=-range[1], high=range[1])
        transform = np.float32([[1, 0, xdelta], [0, 1, ydelta]])
        height, width = img.shape[:2]
        return cv2.warpAffine(img, transform, (width, height))
    def random_shadow(img):
        height, width = img.shape[:2]
        x1, y1 = np.random.uniform(high=width), 0
        x2, y2 = np.random.uniform(high=width), height
        xm, ym = np.mgrid[0:height, 0:width]
        mask = np.zeros_like(img[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.4, high=0.8)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    def random_brightness(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ratio = 1.0 + 0.2 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = random_flip(img)
    img = random_translate(img)
    img = random_shadow(img)
    img = random_brightness(img)
    if rgb:
        return img, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess_mask(img, rgb=False, hsv_min=HSV_MIN, hsv_max=HSV_MAX):
    # Convert color format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Color Mask
    img = mask_function(img, hsv_min, hsv_max)
    # Done
    if rgb:
        return img, np.transpose(np.array([img] * 3), [1, 2, 0])
    return img

def preprocess_leorover(img, rgb=False, hsv_min=HSV_MIN, hsv_max=HSV_MAX):
    # Convert color format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Crop
    img = img[CROP_TOP:, :, :]
    # Color Mask
    img = mask_function(img, hsv_min, hsv_max)
    img = img.astype(np.float32)
    # Resize
    img = cv2.resize(img, (LF_IMAGE_WIDTH, LF_IMAGE_HEIGHT))
    # Normalize
    img = img / 255.0
    # Done
    if rgb:
        return img[:, :, np.newaxis], np.transpose(np.array([img] * 3), [1, 2, 0])
    return img[:, :, np.newaxis]

def preprocess_selforacle_mask(img, rgb=False, hsv_min=HSV_MIN, hsv_max=HSV_MAX):
    # Convert color format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Crop
    img = img[CROP_TOP:, :, :]
    # Color Mask
    img = mask_function(img, hsv_min, hsv_max)
    # Resize
    img = cv2.resize(img, (SO_IMAGE_WIDTH, SO_IMAGE_HEIGHT), cv2.INTER_AREA)
    # Normalize: This should already be done by AnomalyDetector.normalize_and_reshape
    #img = img / 255.0
    img_onechannel = img
    img = np.transpose(np.array([img] * SO_IMAGE_CHANNELS), [1, 2, 0])
    # Done
    if rgb:
        img_rgb = img
        if SO_IMAGE_CHANNELS != 3:
            img_rgb = img_onechannel / 255.0
            img_rgb = np.transpose(np.array([img_rgb] * 3), [1, 2, 0])
        return img, img_rgb
    return img

def preprocess_selforacle(img, rgb=False):
    # Convert color format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # Crop
    img = img[CROP_TOP:, :, :]
    # Resize
    img = cv2.resize(img, (SO_IMAGE_WIDTH, SO_IMAGE_HEIGHT), cv2.INTER_AREA)
    # Normalize: This should already be done by AnomalyDetector.normalize_and_reshape
    #img = img / 255.0
    # Done
    if rgb:
        return img, cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    return img

def preprocess_selforacle_dave2(img, rgb=False):
    # Convert color format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # Crop
    img = img[60:-25, :, :] # remove the sky and the car front
    # Resize
    img = cv2.resize(img, (SO_IMAGE_WIDTH, SO_IMAGE_HEIGHT), cv2.INTER_AREA)
    # Normalize: This should already be done by AnomalyDetector.normalize_and_reshape
    #img = img / 255.0
    # Done
    if rgb:
        return img, cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    return img

def preprocess_dave2(img, rgb=False, size=(DAVE2_IMAGE_WIDTH, DAVE2_IMAGE_HEIGHT)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = img[60:-25, :, :]  # remove the sky and the car front
    img = cv2.resize(img, size, cv2.INTER_AREA)
    img = img.astype('float32')
    if rgb:
        return img, cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    return img

IMG_PROCESSING = {
    'null': preprocess_null,
    'bgr': preprocess_bgr,
    'rgb': preprocess_rgb,
    'crop': preprocess_crop,
    'augment': preprocess_augment,
    'mask': preprocess_mask,
    'leorover': preprocess_leorover,
    'selforacle': preprocess_selforacle,
    'selforacle_mask': preprocess_selforacle_mask,
    'selforacle_dave2': preprocess_selforacle_dave2,
    'dave2': preprocess_dave2,
    'dave2large': lambda img, rgb: preprocess_dave2(img=img, rgb=rgb, size=(DAVE2_IMAGE_WIDTH2, DAVE2_IMAGE_HEIGHT2)),
}

def process_image(image, processing=None, rgb=False):
    if processing:
        if type(processing) is str:
            processing = IMG_PROCESSING[processing]
        image = processing(image, rgb)
    return image

def preprocess_random_augmentation(processing, augmentation_rate):
    def _preprocess(img, rgb=False):
        if np.random.uniform() < augmentation_rate:
            img = preprocess_augment(img=img, rgb=False)
        return process_image(image=img, processing=processing, rgb=rgb)
    return _preprocess

def load_image(image_path, processing=None, rgb=False):
    image = cv2.imread(image_path)
    return process_image(image=image, processing=processing, rgb=rgb)

def save_image(image_path, img, **kwargs):
    return cv2.imwrite(filename=image_path, img=img, **kwargs)

def get_images(data_dir, images_filter=lambda f: f.endswith('.jpg')):
    return map(lambda f: join(data_dir, f), filter(images_filter, listdir(data_dir)))
