'''
Adapted from https://github.com/tsigalko18/transferability-testing-sdcs/blob/main/visualodometry/corruptions/hendrycks.py
Credits: Stocco et al.
@article{2022-Stocco-TSE,
  author    = {Andrea Stocco and Brian Pulfer and Paolo Tonella},
  title     = {{Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems}},
  journal   = {IEEE Transactions on Software Engineering},
  year      = {2022},
  url       = {https://ieeexplore.ieee.org/document/9869302},
  publisher = {IEEE}
}
-----------------------------------------------------------------------------------------------------------
Adapted from https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py
Credits: Dan Hendrycks
@article{hendrycks2019robustness,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Dan Hendrycks and Thomas Dietterich},
  journal={Proceedings of the International Conference on Learning Representations},
  year={2019}
}
'''

# /////////////// Distortion Helpers ///////////////

from math import ceil, log2
import numpy as np
import skimage as sk
from skimage.filters import gaussian
import cv2
from scipy.ndimage import zoom as scizoom
import warnings

warnings.simplefilter("ignore", UserWarning)


def auc(errs):  # area under the alteration error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h, w = img.shape[0], img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))
    cw = int(np.ceil(w / zoom_factor))

    top = (h - ch) // 2
    left = (w - cw) // 2
    img = scizoom(img[top:top + ch, left:left + cw], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_left = (img.shape[1] - w) // 2

    return img[trim_top:trim_top + h, trim_left:trim_left + w]


# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.

    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


# def fgsm(x, source_net, severity=1):
#     c = [8, 16, 32, 64, 128][severity - 1]
#
#     x = V(x, requires_grad=True)
#     logits = source_net(x)
#     source_net.zero_grad()
#     loss = F.cross_entropy(logits, V(logits.data.max(1)[1].squeeze_()), size_average=False)
#     loss.backward()
#
#     return standardize(torch.clamp(unstandardize(x.data) + c / 255. * unstandardize(torch.sign(x.grad.data)), 0, 1))


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=2)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=2) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(224 - c[1], c[1], -1):
            for w in range(224 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=2), 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def fog(x, severity=1):
    width, height, chan = x.shape
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1], mapsize=2**ceil(log2(max(width, height))))[:width, :height][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    #c = [.1, .2, .3, .4, .5][severity - 1]
    #x = np.array(x) / 255.
    #x = sk.color.rgb2hsv(x)
    #x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 255)
    #x = sk.color.hsv2rgb(x)
    #return np.clip(x, 0, 1) * 255
    c = [25, 50, 75, 100, 125][severity -1]
    x = np.array(x, dtype=np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    x[:,:,2] = x[:,:,2] + c 
    x[:,:,2][x[:,:,2]>255] = 255 
    x = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
    return x


def saturate(x, severity=1):
    #c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    #x = np.array(x) / 255.
    #x = sk.color.rgb2hsv(x)
    #x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    #x = sk.color.hsv2rgb(x)
    #return np.clip(x, 0, 1) * 255
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.
    x = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    x[:,:,1] = x[:,:,1] * c[0] + c[1] 
    x[:,:,1][x[:,:,1]>1.] = 1. 
    x = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
    return x * 255.

def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    retval, encoded = cv2.imencode(ext='.jpg', img=x, params=[cv2.IMWRITE_JPEG_QUALITY, c])
    x = cv2.imdecode(buf=encoded, flags=cv2.IMREAD_UNCHANGED)

    return x


def pixelate(x, severity=1):
    # start_time = time.time_ns()
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    width, height, chan = x.shape
    width2, height2 = int(width * c), int(height * c)
    x = cv2.resize(x, dsize=(height2, width2), interpolation=cv2.INTER_NEAREST)
    x = cv2.resize(x, dsize=(height, width), interpolation=cv2.INTER_NEAREST)
    # print("%s ns" % (time.time_ns() - start_time))

    return x

# /////////////// End Distortions ///////////////


# /////////////// Functions Dict ///////////////

DISTORTIONS = {
    'GaussianNoise': gaussian_noise,
    'ShotNoise': shot_noise,
    'ImpulseNoise': impulse_noise,
    'DefocusBlur': defocus_blur,
    'GlassBlur': glass_blur,
    'Fog': fog,
    'Brightness': brightness,
    'Contrast': contrast,
    'Pixelate': pixelate,
    'JPEG': jpeg_compression,
    'SpeckleNoise': speckle_noise,
    'GaussianBlur': gaussian_blur,
    'Spatter': spatter,
    'Saturate': saturate,
}

def get_anomaly(name, severity: int=1):
    return lambda img, severity=severity: DISTORTIONS[name](img, severity=severity).astype(np.uint8)

# /////////////// End Functions Dict ///////////////
