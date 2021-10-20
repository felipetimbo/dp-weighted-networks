import numpy as np

variance = 100000

def geom_prob_mass(eps, sensitivity=1):
    a = np.exp(-eps/sensitivity)
    weights = np.array(range(-variance, variance))
    prob_mass = []
    for w in weights:
        m = ((1 - a)/(1 + a))*(a** np.absolute(w) )
        prob_mass.append(m)
    return prob_mass

def geometric(arr, prob_mass):
    noisy_arr = arr + np.random.choice(a=len(prob_mass), size=len(arr), p=prob_mass, replace=True) - variance
    return np.array(noisy_arr)

def geometric_mechanism(arr, eps, sensitivity=1):
    prob_mass = geom_prob_mass(eps, sensitivity)
    noisy_arr = arr + np.random.choice(a=len(prob_mass), size=len(arr), p=prob_mass, replace=True) - variance
    return np.array(noisy_arr)