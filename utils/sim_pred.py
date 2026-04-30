import numpy as np


def pwlm_mfcc(mfcc_dist):
    d0 = 0.464
    m = -69.4
    mfcc_dist = np.log10(mfcc_dist)
    if mfcc_dist <= d0:
        rating = 100
    else:
        rating = 100 + m*(mfcc_dist - d0)
    return rating