import numpy as np
import mpmath 
import multiprocessing

from dpwnets import utils
from random import randrange
from scipy.linalg import convolution_matrix

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

def log_laplace_mechanism(arr, eps, sensitivity=2):
    lap = np.random.laplace(loc=0, scale=sensitivity/eps, size=len(arr))
    #  noisy_arr = np.around(np.clip( np.log(arr) + lap, 1, None )).astype(int)
    noisy_arr = np.around(np.clip( arr * np.exp(lap) , 1, max(arr) )).astype(int)  # https://arxiv.org/pdf/2101.02957.pdf

    return np.array(noisy_arr)

def find_point_of_decreasing(m, e):
    # m = len(arr)

    size_level_l_before = 0
    for l in range(37235520800000, 100000000000000):
        if l % 100000 == 0:
            utils.log_msg('level: %s' % str(l))
        size_level_l_up = mpmath.binomial(l+m-1,m-1)
        size_level_l_up_exp = mpmath.exp( mpmath.fprod( [mpmath.mpf(e), mpmath.fneg(mpmath.mpf(l)), mpmath.mpf(0.5)]) )
        size_level_l = mpmath.fmul( size_level_l_up_exp, size_level_l_up )
        if size_level_l < size_level_l_before:
            break
        else: 
            size_level_l_before = size_level_l
    
    return l

def get_levels_size(m, levels_range, num_threads=1):
    # m = len(arr)
    # non_zero_arr = np.array(arr)[np.nonzero(arr)[0]]
    
    # levels_size_up = []
    # levels_size_down = [ mpmath.mpf(1.0)]
    
    range_l = np.array_split(list(range(levels_range[0], levels_range[1])), num_threads)

    manager = multiprocessing.Manager()
    levels_size_multiprocessing = manager.dict() 

    threads = []
    for i in range(num_threads):
        t = multiprocessing.Process( target = compute_levels_size_parallel, args =( range_l[i], m, levels_size_multiprocessing ))
        t.start()
        threads.append(t)
        
    for t in threads:    
        t.join()

    utils.log_msg('transforming runs in a dictionary...')
    levels_size_dict = dict(levels_size_multiprocessing) 

    utils.log_msg('sorting levels...')
    levels_size = np.array([levels_size_dict[key] for key in sorted(levels_size_dict.keys())])


    #     if l < len(non_zero_arr):
    #         w = arr[l] # weight of element at position l 
    #         num_columns_of_levels_size_down = len(levels_size_down) + w 
    #         a = convolution_matrix(levels_size_down, num_columns_of_levels_size_down, mode='valid')
    #         levels_size_down = a.sum(axis=0)

    # if len(levels_size_down) < len(levels_size_up):
    #     levels_size_down = np.append(levels_size_down, [0] * (len(levels_size_up) - len(levels_size_down)))
    # elif len(levels_size_down) > len(levels_size_up):
    #     levels_size_down = levels_size_down[:levels_len] #np.append(levels_size_up, [0] * (len(levels_size_down) - len(levels_size_up)))

    # levels_size = levels_size_up # + levels_size_down
    return levels_size

def compute_probability_mass(levels_range, levels_size, e):

    neg = np.vectorize(mpmath.fneg)
    mul = np.vectorize(mpmath.fmul)
    exp = np.vectorize(mpmath.exp)

    utils.log_msg('computing probability mass...')
    sequence = mpmath.arange(levels_range[0], levels_range[1])
    sequence_negatives = neg(sequence)
    sequence_negatives_times_e = mul(sequence_negatives, mpmath.mpf(e))
    sequence_negatives_div_two = mul(sequence_negatives_times_e, mpmath.mpf(0.5))
    exponential = exp( sequence_negatives_div_two )

    prob_mass = exponential * levels_size

    utils.log_msg('normalizing pr mass...')
    sum_prob_mass = mpmath.fsum(prob_mass)
    normalized_prob_mass = []
    for mass in prob_mass:
        normalized_prob_mass.append(mpmath.fdiv(mass, sum_prob_mass))
    
    return normalized_prob_mass

def exponential_mechanism(levels_range, normalized_prob_mass):
    return np.random.choice(list(range(levels_range[0], levels_range[1])), size=1, p=normalized_prob_mass)[0]

def compute_levels_size_parallel(levels_range, m, levels_size_mult):

    for l in range(levels_range[0], levels_range[-1]+1):
        size_level_l_up = mpmath.binomial(l+m-1,m-1)
        levels_size_mult[l] = size_level_l_up

    utils.log_msg('thread finished' )

if __name__ == "__main__":
    arr = [ 3, 1, 2, 0, 0, 0 ]
    get_levels_size(arr)