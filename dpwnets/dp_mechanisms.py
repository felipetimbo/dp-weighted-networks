import numpy as np
import mpmath 

from random import randrange
from scipy.linalg import convolution_matrix

np.random.seed(1)

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

def log_laplace_mechanism(arr, eps, sensitivity):
    lap = np.random.laplace(loc=0, scale=sensitivity/eps, size=len(arr))
    #  noisy_arr = np.around(np.clip( np.log(arr) + lap, 1, None )).astype(int)
    noisy_arr = np.around(np.clip( arr * np.exp(lap) , 1, max(arr) )).astype(int)  # https://arxiv.org/pdf/2101.02957.pdf

    return np.array(noisy_arr)

def get_levels_size(arr):
    m = len(arr)
    non_zero_arr = np.array(arr)[np.nonzero(arr)[0]]
    levels_len = 100000
    
    levels_size_up = []
    levels_size_down = [ mpmath.mpf(1.0)]
    
    for l in range(levels_len+1):
        size_level_l_up = mpmath.binomial(l+m-1,m-1)
        levels_size_up.append(size_level_l_up)

        if l < len(non_zero_arr):
            w = arr[l] # weight of element at position l 
            num_columns_of_levels_size_down = len(levels_size_down) + w 
            a = convolution_matrix(levels_size_down, num_columns_of_levels_size_down, mode='valid')
            levels_size_down = a.sum(axis=0)

        # else:
        #     levels_size_down = np.append(levels_size_down, 0)

    if len(levels_size_down) < len(levels_size_up):
        levels_size_down = np.append(levels_size_down, [0] * (len(levels_size_up) - len(levels_size_down)))
    elif len(levels_size_down) > len(levels_size_up):
        # levels_size_up = np.append(levels_size_up, [0] * (len(levels_size_down) - len(levels_size_up)))
        levels_size_down = levels_size_down[:levels_len] #np.append(levels_size_up, [0] * (len(levels_size_down) - len(levels_size_up)))

    levels_size = levels_size_up + levels_size_down
    return levels_size_up
    # descomentar bibliotecas

def compute_probability_mass(levels_size, e):
    # prob_mass = []
    # total_mass = 0

    neg = np.vectorize(mpmath.fneg)
    mul = np.vectorize(mpmath.fmul)
    exp = np.vectorize(mpmath.exp)

    sequence = mpmath.arange(1, len(levels_size))
    sequence_negatives = neg(sequence)
    sequence_negatives_times_e = mul(sequence_negatives, mpmath.mpf(e))
    sequence_negatives_div_two = mul(sequence_negatives_times_e, mpmath.mpf(0.5))
    exponential = exp( sequence_negatives_div_two )

    prob_mass = exponential * levels_size[1:]
    
    return prob_mass

def exponential_mechanism(prob_mass):
    # div = np.vectorize(mpmath.fdiv)

    sum_prob_mass = mpmath.fsum(prob_mass)
    # normalized_prob_mass = div(prob_mass, sum_prob_mass)

    normalized_prob_mass = []
    for mass in prob_mass:
        normalized_prob_mass.append(mpmath.fdiv(mass, sum_prob_mass))

    return np.random.choice(a=len(prob_mass), size=1, p=normalized_prob_mass), normalized_prob_mass

if __name__ == "__main__":
    arr = [ 3, 1, 2, 0, 0, 0 ]
    get_levels_size(arr)