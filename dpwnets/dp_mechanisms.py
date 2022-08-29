import numpy as np
from random import randrange

variance = 100000

np.random.seed(1)

def geom_prob_mass(eps, sensitivity=1):
    a = np.exp(-eps/sensitivity)
    weights = np.array(range(-variance, variance))
    prob_mass = []
    for w in weights:
        m = ((1 - a)/(1 + a))*(a** np.absolute(w) )
        prob_mass.append(m)
    return prob_mass

def geometric(arr, prob_mass):
    # np.random.seed(3)
    noisy_arr = arr + np.random.choice(a=len(prob_mass), size=len(arr), p=prob_mass, replace=True) - variance
    return np.array(noisy_arr)

def geometric2(arr, prob_mass, seed=0):
    x = randrange(100000000)
    np.random.seed( x )
    noisy_arr = arr + np.random.choice(a=len(prob_mass), size=len(arr), p=prob_mass, replace=True) - variance
    return np.array(noisy_arr)

def geometric_mechanism(arr, eps, sensitivity=1):
    p = 1 - np.exp(-eps/sensitivity)
    z = np.random.geometric(p, len(arr)) - np.random.geometric(p, len(arr))
    noisy_arr = arr + z
    return np.array(noisy_arr)