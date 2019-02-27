import math
import numpy as np
import scipy

def ackley(x):
    x = np.atleast_2d(x)
    d = x.shape[1]

    c = 2*math.pi
    b = 0.2
    a = 20

    sum1 = 0
    sum2 = 0
    for ii in range(d):
        xi = x[:, ii]
        sum1 = sum1 + xi**2
        sum2 = sum2 + np.cos(c*xi)

    term1 = -a * np.exp(-b*np.sqrt(sum1/d))
    term2 = -np.exp(sum2/d)

    y = term1 + term2 + a + np.exp(1)
    return -y

def gaussian_2d(x, mu=0.75, sigma=0.125):
    return np.exp(-sum((x-mu)**2/(2*sigma**2)))

def add_nonuniform_noise(field, noise_level):
    epsilon = np.random.uniform(-noise_level, noise_level, field.shape)
    amplitude = np.ones(field.shape)
    for row in range(amplitude.shape[0]):
        y = row / amplitude.shape[0]
        for j in range(amplitude):
            x = col / amplitude.shape[1]
            amplitude[row, col] = gaussian_2d(np.array([x, y]))
    return field + amplitude*epsilon

def add_uniform_noise(field, noise_level):
    epsilon = np.random.uniform(-noise_level, noise_level, field.shape)
    return field + epsilon