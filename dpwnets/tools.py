import math
from xxlimited import new
import pandas as pd
import numpy as np
import graph_tool as gt
import cvxpy as cp
import graph_tool.spectral as sp
import os 
import itertools
import mpmath

import matplotlib
import matplotlib.pyplot as plt

from scipy.linalg import convolution_matrix

from dpwnets import (utils, graphics)
from graph.wgraph import WGraph
from graph_tool.util import find_edge
from graph_tool.generation import random_graph

np.random.seed(0)

def high_pass_filter(arr, eps, max_len, threshold):

    a = np.exp(-eps)
    theta = math.log( ((1+a)*threshold)/(max_len - threshold)) / math.log(a)
    
    non_zero_edges_w_filtered_mask = arr >= theta
    non_zero_edges_w_filtered_pos = np.where(non_zero_edges_w_filtered_mask)[0]
    
    num_remaining_edges = np.sum( non_zero_edges_w_filtered_mask )
    
    p = (a**theta)/(1 + a)
    len_binomial = max_len - threshold
    while True:
        k = np.random.binomial( len_binomial, p)
    
        num_edges_after_filter = num_remaining_edges + k
        num_exceding_edges = int(num_edges_after_filter - threshold)
    
        if num_exceding_edges > 0:
            break
            # utils.error_msg('size of high pass filter less than desired')
    
    non_zero_edges_w_filtered = arr[non_zero_edges_w_filtered_mask]
    zeros_edges_w_filtered = np.array(list(map(lambda u: math.log( (1 - u)/a, a) + theta + 1, np.random.uniform(0, 1, k))))
    top_edges_after_filter = np.append(non_zero_edges_w_filtered, zeros_edges_w_filtered)
    
    m_pos = top_edges_after_filter.argsort()[:num_exceding_edges]
    non_zero_edges_w_to_be_removed = m_pos[m_pos < num_remaining_edges]
    if len(non_zero_edges_w_to_be_removed)>0:
        non_zero_edges_w_filtered_mask[non_zero_edges_w_filtered_pos[non_zero_edges_w_to_be_removed]] = False
        # utils.log_msg('%s non zero edges removed' % len(non_zero_edges_w_to_be_removed))
        num_remaining_edges -= len(non_zero_edges_w_to_be_removed)
    
    zero_edges_w_to_be_removed = m_pos[m_pos >= num_remaining_edges]    
    # utils.log_msg('%s zero edges removed' % len(zero_edges_w_to_be_removed))          
    
    top_edges_after_filter_mask = np.full(len(top_edges_after_filter), True)
    top_edges_after_filter_mask[non_zero_edges_w_to_be_removed] = False
    top_edges_after_filter_mask[zero_edges_w_to_be_removed] = False
    top_m_edges_w_noisy = top_edges_after_filter[top_edges_after_filter_mask]
    
    # utils.log_msg('m = %s/%s, orig. remaining edges = %s/%s (%s) ' % ( len(top_m_edges_w_noisy), len(arr), num_remaining_edges, len(arr), float("{:.2f}".format(num_remaining_edges/len(arr))) ) )

    return top_m_edges_w_noisy, num_remaining_edges, non_zero_edges_w_filtered_mask

def threshold_sampling(arr, eps, max_len, m):

    Ts = 2

    variance = 100000
    a = np.exp(-eps)

    len_binomial = max_len - m

    while True:
        p = (a/((1+a)*Ts))*((1-(Ts+1)*(a**Ts) + Ts*(a**(Ts+1)))/(1-a)) + (a**(Ts+1))/(1+a)
        k = np.random.binomial( len_binomial, p)
        if k > m:
            Ts += 1
        # num_edges = np.sum( P > Ts ) + k
        # if num_edges > 1.1*m:
        #     Ts += 1
        else:
            # Ts -= 1
            break

    Ts_before = Ts
    p_before = p
    k_before = k
    edges_added_in_G_prime_pos_before = []

    while True:
        p = (a/((1+a)*Ts))*((1-(Ts+1)*(a**Ts) + Ts*(a**(Ts+1)))/(1-a)) + (a**(Ts+1))/(1+a)
        k = np.random.binomial( len_binomial, p)

        edges_added_in_G_prime_pos = []
        for edge in arr:
            prob_added = max(min(edge/Ts, 1), 0)
            is_added = np.random.choice([True, False], 1, p=[prob_added, 1-prob_added])[0]
            edges_added_in_G_prime_pos.append(is_added)

        total_new_edges = k + np.sum(edges_added_in_G_prime_pos)
        if total_new_edges > m:
            Ts_before = Ts
            p_before = p
            k_before = k
            edges_added_in_G_prime_pos_before = edges_added_in_G_prime_pos.copy()
            Ts += 1
        else:
            # Ts -= 1
            break

    Ts = Ts_before
    p = p_before
    k = k_before
    edges_added_in_G_prime_pos = np.array(edges_added_in_G_prime_pos_before)

    weights = np.array(range(variance))
    prob_mass = []
    for w in weights:
        if w <= Ts:
            mass = ((w/Ts)*((1-a)*(a**(abs(w))))/(1+a))/p
        else:
            mass = (((1-a)*(a**(abs(w))))/(1+a))/p
        prob_mass.append(mass)

    zeros_edges_w_filtered = np.random.choice(list(range(variance)), k, p=prob_mass, replace=True ) 
    if np.sum(edges_added_in_G_prime_pos) > 0:   
        non_zero_edges_w_filtered = arr[edges_added_in_G_prime_pos]
    else:
        non_zero_edges_w_filtered = np.array([])

    top_edges_after_filter = np.append(non_zero_edges_w_filtered, zeros_edges_w_filtered)

    num_remaining_edges = np.sum( edges_added_in_G_prime_pos )

    num_exceding_edges = int(num_remaining_edges + k - m)
    m_pos = top_edges_after_filter.argsort()[:num_exceding_edges]
    non_zero_edges_w_to_be_removed = m_pos[m_pos < num_remaining_edges]
    non_zero_edges_w_filtered_pos = np.where(edges_added_in_G_prime_pos)[0]

    if len(non_zero_edges_w_to_be_removed)>0:
        edges_added_in_G_prime_pos[non_zero_edges_w_filtered_pos[non_zero_edges_w_to_be_removed]] = False
        # utils.log_msg('%s non zero edges removed' % len(non_zero_edges_w_to_be_removed))
        num_remaining_edges -= len(non_zero_edges_w_to_be_removed)
    
    zero_edges_w_to_be_removed = m_pos[m_pos >= num_remaining_edges]    
    # utils.log_msg('%s zero edges removed' % len(zero_edges_w_to_be_removed))          
    
    top_edges_after_filter_mask = np.full(len(top_edges_after_filter), True)
    top_edges_after_filter_mask[non_zero_edges_w_to_be_removed] = False
    top_edges_after_filter_mask[zero_edges_w_to_be_removed] = False
    top_m_edges_w_noisy = top_edges_after_filter[top_edges_after_filter_mask]
    
    # utils.log_msg('m = %s/%s, orig. remaining edges = %s/%s (%s) ' % ( len(top_m_edges_w_noisy), len(arr), num_remaining_edges, len(arr), float("{:.2f}".format(num_remaining_edges/len(arr))) ) )

    return top_m_edges_w_noisy, num_remaining_edges, edges_added_in_G_prime_pos

    # edges_after_filter = np.append(arr, zeros_edges_w_filtered)

    # rand_values = np.random.uniform(0,1,len(zeros_edges_w_filtered))
    # P = np.append(P, zeros_edges_w_filtered/rand_values)
    
    # CONTINUAR AQUI

    # m_highest_priorities_mask = P.argsort()[::-1][:m]

    # non_zero_edges_w_to_keep_pos = np.sort(m_highest_priorities_mask[m_highest_priorities_mask < len(arr)])
    # zero_edges_w_to_keep_pos = np.sort(m_highest_priorities_mask[m_highest_priorities_mask >= len(arr)])

    # top_edges_after_filter = np.append(edges_after_filter[non_zero_edges_w_to_keep_pos], edges_after_filter[zero_edges_w_to_keep_pos])
    # if min(top_edges_after_filter) <= 0:
    #     utils.error_msg('negative values in priority sampling')

    # non_zero_edges_w_filtered_mask = np.full(len(arr), False)
    # non_zero_edges_w_filtered_mask[non_zero_edges_w_to_keep_pos] = True

    # num_remaining_edges = np.sum(non_zero_edges_w_filtered_mask)

    # return top_edges_after_filter, num_remaining_edges, non_zero_edges_w_filtered_mask

def priority_sampling(arr, eps, max_len, m):

    Ts = 2

    variance = 100000
    a = np.exp(-eps)

    len_binomial = max_len - m

    while True:
        p = (a/((1+a)*Ts))*((1-(Ts+1)*(a**Ts) + Ts*(a**(Ts+1)))/(1-a)) + (a**(Ts+1))/(1+a)
        k = np.random.binomial( len_binomial, p)
        if k > m:
            Ts += 1
        # num_edges = np.sum( P > Ts ) + k
        # if num_edges > 1.1*m:
        #     Ts += 1
        else:
            Ts -= 1
            break

    p = (a/((1+a)*Ts))*((1-(Ts+1)*(a**Ts) + Ts*(a**(Ts+1)))/(1-a)) + (a**(Ts+1))/(1+a)
    k = np.random.binomial( len_binomial, p)

    rand_values = np.random.uniform(0,1,len(arr))
    P = np.clip(arr/rand_values, 0, None)    # Priorities

    weights = np.array(range(variance))
    prob_mass = []
    for w in weights:
        if w <= Ts:
            mass = ((w/Ts)*((1-a)*(a**(abs(w))))/(1+a))/p
        else:
            mass = (((1-a)*(a**(abs(w))))/(1+a))/p
        prob_mass.append(mass)

    zeros_edges_w_filtered = np.random.choice(list(range(variance)), k, p=prob_mass, replace=True )    
    edges_after_filter = np.append(arr, zeros_edges_w_filtered)

    rand_values = np.random.uniform(0,1,len(zeros_edges_w_filtered))
    P = np.append(P, zeros_edges_w_filtered/rand_values)
    
    m_highest_priorities_mask = P.argsort()[::-1][:m]

    non_zero_edges_w_to_keep_pos = np.sort(m_highest_priorities_mask[m_highest_priorities_mask < len(arr)])
    zero_edges_w_to_keep_pos = np.sort(m_highest_priorities_mask[m_highest_priorities_mask >= len(arr)])

    top_edges_after_filter = np.append(edges_after_filter[non_zero_edges_w_to_keep_pos], edges_after_filter[zero_edges_w_to_keep_pos])
    if min(top_edges_after_filter) <= 0:
        utils.error_msg('negative values in priority sampling')

    non_zero_edges_w_filtered_mask = np.full(len(arr), False)
    non_zero_edges_w_filtered_mask[non_zero_edges_w_to_keep_pos] = True

    num_remaining_edges = np.sum(non_zero_edges_w_filtered_mask)

    return top_edges_after_filter, num_remaining_edges, non_zero_edges_w_filtered_mask

def priority_sampling_old(arr, eps, max_len, m):

    Ts = 2

    variance = 100000
    a = np.exp(-eps)

    len_binomial = max_len - m

    while True:
        p = (a/((1+a)*Ts))*((1-(Ts+1)*(a**Ts) + Ts*(a**(Ts+1)))/(1-a)) + (a**(Ts+1))/(1+a)
        k = np.random.binomial( len_binomial, p)
        if k > m:
            Ts += 1
        # num_edges = np.sum( P > Ts ) + k
        # if num_edges > 1.1*m:
        #     Ts += 1
        else:
            Ts -= 1
            break

    p = (a/((1+a)*Ts))*((1-(Ts+1)*(a**Ts) + Ts*(a**(Ts+1)))/(1-a)) + (a**(Ts+1))/(1+a)
    k = np.random.binomial( len_binomial, p)

    rand_values = np.random.uniform(0,1,len(arr))
    P = np.clip(arr/rand_values, 0, None)    # Priorities

    weights = np.array(range(variance))
    prob_mass = []
    for w in weights:
        if w <= Ts:
            mass = ((w/Ts)*((1-a)*(a**(abs(w))))/(1+a))/p
        else:
            mass = (((1-a)*(a**(abs(w))))/(1+a))/p
        prob_mass.append(mass)

    zeros_edges_w_filtered = np.random.choice(list(range(variance)), k, p=prob_mass, replace=True )    
    edges_after_filter = np.append(arr, zeros_edges_w_filtered)

    rand_values = np.random.uniform(0,1,len(zeros_edges_w_filtered))
    P = np.append(P, zeros_edges_w_filtered/rand_values)
    
    m_highest_priorities_mask = P.argsort()[::-1][:m]

    non_zero_edges_w_to_keep_pos = np.sort(m_highest_priorities_mask[m_highest_priorities_mask < len(arr)])
    zero_edges_w_to_keep_pos = np.sort(m_highest_priorities_mask[m_highest_priorities_mask >= len(arr)])

    top_edges_after_filter = np.append(edges_after_filter[non_zero_edges_w_to_keep_pos], edges_after_filter[zero_edges_w_to_keep_pos])
    if min(top_edges_after_filter) <= 0:
        utils.error_msg('negative values in priority sampling')

    non_zero_edges_w_filtered_mask = np.full(len(arr), False)
    non_zero_edges_w_filtered_mask[non_zero_edges_w_to_keep_pos] = True

    num_remaining_edges = np.sum(non_zero_edges_w_filtered_mask)

    return top_edges_after_filter, num_remaining_edges, non_zero_edges_w_filtered_mask

def gradient_descent(init, steps, grad, laplaced_sum, proj=lambda x: x, min_value=1):
    xs = [init]
    for step in steps:
        xs_next = proj(xs[-1] - step * grad(xs[-1]), laplaced_sum, min_value)
        xs = [xs_next]
    return xs

def gradient_descent2(init, steps, grad, proj=lambda x: x):
    xs = [init]
    for step in steps:
        xs.append( proj( xs[-1] - step * grad(xs[-1])) )
    return xs

def l2(x, y):
    return np.sum(np.power( ( y - x ),2) )

def least_squares(A, b, x, num_lines):
    """Least squares objective."""
    return (0.5/num_lines) * np.linalg.norm(A.dot(x)-b)**2

def l2_gradient(x, y):
    return 2*(y - x)

def least_squares_gradient(A, b, x, num_lines):
    """Gradient of least squares objective at x."""
    return A.T.dot(A.dot(x)-b)/num_lines

def proj(x, desired_sum, min_value=1):
    """Projection of x onto the subspace"""
    current_sum = np.sum(x)
    factor = (current_sum - desired_sum)/len(x)
    projection = np.around(np.clip(x - factor, min_value, None))     
    
    return projection

def proj2(x, min_value=1):
    """Projection of x onto the subspace"""
    projection = np.around(np.clip(x, min_value, None))     

    return projection

def min_l2_norm_bkp(edges_w, desired_sum, num_steps=500, min_value=1):
    x0 = edges_w.copy()

    alpha = 0.01
    gradient = lambda y: l2_gradient(edges_w, y)
    xs = gradient_descent(x0, [alpha]*num_steps, gradient, desired_sum, proj, min_value)
    new_edges_w = xs[-1].astype(int)

    exceding_units = int(np.abs(desired_sum - np.sum(new_edges_w)))
    if exceding_units != 0:
        edges_w_different_1_pos = np.where(new_edges_w != min_value)[0]

        if len(edges_w_different_1_pos) >= exceding_units:
            edges_w_picked = edges_w_different_1_pos[np.random.choice(len(edges_w_different_1_pos), exceding_units, replace=False)]
            if (desired_sum - np.sum(new_edges_w)) < 0:
                new_edges_w[edges_w_picked] -= 1
            else: 
                new_edges_w[edges_w_picked] += + 1

    return new_edges_w

def min_l2_norm_old(arr_orig, desired_sum, num_steps=500, min_value=1):
    arr = np.array(arr_orig).astype('int')

    if desired_sum < len(arr)*min_value:
        return (np.ones(len(arr))*min_value).astype('int')
    # higher_than_min_idx = np.where(arr > min_value)[0]
    # lower_than_min_idx = np.where(arr <= min_value)[0]

    arr = np.clip(arr, min_value, np.max(arr))
    
    # less_than_min_arr = arr[arr < min_value]
    # less_than_min_arr_sum = np.sum(np.abs(less_than_min_arr))
    exceding_units = int(np.sum(arr) -desired_sum)

    # higher_than_min_arr_sum = np.sum(np.abs(less_than_min_arr))

    if exceding_units > 0:
        sign = -1
    elif exceding_units < 0:
        sign = 1
    else:
        sign = 0

    exceding_units = np.abs(exceding_units)

    if sign == -1:

        while exceding_units > len(np.where(arr > min_value)[0]):
            exceding_units -= len(arr[np.where(arr > min_value)[0]])
            arr[np.where(arr > min_value)[0]] += sign*1
        if exceding_units > 0:
            p = np.ones(len(arr)).astype('int')
            p[np.where(arr <= min_value)[0]] = 0
            prob = p/np.sum(p)
            idx_to_decrease_1 = np.random.choice(len(arr), exceding_units, replace=False, p=prob)
            arr[idx_to_decrease_1] += sign*1
    elif sign == 1:
        while exceding_units > len(arr):
            exceding_units -= len(arr)
            arr += sign*1
        if exceding_units > 0:
            idx_to_increase_1 = np.random.choice(len(arr), exceding_units, replace=False)
            arr[idx_to_increase_1] += sign*1

        # num_iterations = int(np.ceil(exceding_units/len(arr[higher_than_min_idx])) )
        # if num_iterations > 1:
        #     for _ in range(num_iterations-1):
        #         arr[higher_than_min_idx] -= 1

    
    
    if np.sum(arr) != desired_sum and len(np.where(arr < min_value)[0]) > 0:
        print("error in min l2 method")

    return arr

    # alpha = 0.01
    # gradient = lambda y: l2_gradient(edges_w, y)
    # xs = gradient_descent(x0, [alpha]*num_steps, gradient, desired_sum, proj, min_value)
    # new_edges_w = xs[-1].astype(int)

    # exceding_units = int(np.abs(desired_sum - np.sum(new_edges_w)))
    # if exceding_units != 0:
    #     edges_w_different_1_pos = np.where(new_edges_w != min_value)[0]

    #     if len(edges_w_different_1_pos) >= exceding_units:
    #         edges_w_picked = edges_w_different_1_pos[np.random.choice(len(edges_w_different_1_pos), exceding_units, replace=False)]
    #         if (desired_sum - np.sum(new_edges_w)) < 0:
    #             new_edges_w[edges_w_picked] -= 1
    #         else: 
    #             new_edges_w[edges_w_picked] += + 1

    # return new_edges_w

def min_l2_norm(edges_w, _sum, num_steps=500, min_value=1):
    x0 = edges_w.copy()

    desired_sum = int(max(_sum, min_value * len(edges_w)))

    alpha = 0.01
    gradient = lambda y: l2_gradient(edges_w, y)
    xs = gradient_descent(x0, [alpha]*num_steps, gradient, desired_sum, proj, min_value)
    new_edges_w = xs[-1].astype(int)

    exceding_units = int(np.abs(desired_sum - np.sum(new_edges_w)))

    while exceding_units > 0:
        edges_w_different_1_pos = np.where(new_edges_w != min_value)[0]
        if len(edges_w_different_1_pos) >= exceding_units:
            edges_w_picked = edges_w_different_1_pos[np.random.choice(len(edges_w_different_1_pos), exceding_units, replace=False)]
            exceding_units -= exceding_units
        else:
            edges_w_picked = edges_w_different_1_pos[np.random.choice(len(edges_w_different_1_pos), len(edges_w_different_1_pos), replace=False)]
            exceding_units -= len(edges_w_different_1_pos)

        if (desired_sum - np.sum(new_edges_w)) < 0:
            new_edges_w[edges_w_picked] -= 1
        else: 
            new_edges_w[edges_w_picked] += + 1

    return new_edges_w

def error_plot(ys, yscale='log'):
    plt.figure(figsize=(8, 8))
    plt.xlabel('Step')
    plt.ylabel('Error')
    plt.yscale(yscale)
    plt.plot(range(len(ys)), ys)
    path = "./a.png"
    dir_path = os.path.dirname(os.path.realpath(path))
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(path, dpi=900)

def min_l2_norm2(g, nss, weight=1, num_steps=500):
    
    edges = g.get_edges([g.ep.ew])

    b = np.array(nss)
    n = len(nss)
    m = len(edges)

    num_lines = n+m

    A = np.zeros((num_lines, m), dtype=int)
    # A = np.zeros((n, m), dtype=int)

    for e, i in zip(edges,range(m)):
        A[e[0].astype('int'),e[1].astype('int')] = 1 
        A[e[1].astype('int'),e[0].astype('int')] = 1 
        A[n+i, i] = weight 
        b = np.append(b, e[2] * weight) 

    x0 = edges[:,2]

    alpha = 2.0
    
    objective = lambda x: least_squares(A, b, x, num_lines)
    gradient = lambda x: least_squares_gradient(A, b, x, num_lines)
    xs = gradient_descent2(x0, [alpha]*num_steps, gradient ) #, proj2)
    error_plot([objective(x) for x in xs])
    new_edges_w = xs[-1].astype(int)

    new_edges = np.concatenate((edges[:,(0,1)], np.array([new_edges_w]).T), axis=1)

    return new_edges

def min_l2_norm3(g, nss, weight=1, num_steps=100):
    
    edges = g.get_edges([g.ep.ew])

    b = np.array(nss)
    n = len(nss)
    m = len(edges)

    # num_lines = n+m
    num_lines = n

    A = np.zeros((num_lines, m), dtype=int)
    # A = np.zeros((n, m), dtype=int)

    for e, i in zip(edges,range(m)):
        A[e[0].astype('int'),e[1].astype('int')] = 1 
        A[e[1].astype('int'),e[0].astype('int')] = 1 
        # A[n+i, i] = weight 
         #b = np.append(b, e[2] * weight) 

    x0 = edges[:,2]

    alpha = 0.1
    
    objective = lambda x: least_squares(A, b, x, num_lines)
    gradient = lambda x: least_squares_gradient(A, b, x, num_lines)
    xs = gradient_descent2(x0, [alpha]*num_steps, gradient ) #, proj2)
    error_plot([objective(x) for x in xs])
    new_edges_w = xs[-1].astype(int)

    new_edges = np.concatenate((edges[:,(0,1)], np.array([new_edges_w]).T), axis=1)

    return new_edges


def min_l2_norm_ns(edges, nss, weight=1):

    b = np.array(nss)
    n = len(nss)
    m = len(edges)
    A = np.zeros((n+m, m), dtype=int)
    
    for e, i in zip(edges,range(m)):
        A[e[0].astype('int'),e[1].astype('int')] = 1 
        A[e[1].astype('int'),e[0].astype('int')] = 1 
        A[n+i, i] = weight 
        b = np.append(b, e[2] * weight) 

    x = cp.Variable(m)
    objective = cp.Minimize(cp.sum_squares(A@x - b))
    constraints = [x >= 1 ]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    new_edges_w = np.clip(np.around(x.value), 0, np.max(np.around(x.value)))
    return new_edges_w

def sample_graph(g, edges_w_noisy_non_zero, edges_w_noisy_zeros_pp):

    edges_w_noisy_zeros = edges_w_noisy_zeros_pp[np.nonzero(edges_w_noisy_zeros_pp)]

    new_g = WGraph()
    new_g.add_vertex(n=g.n())
    optin = new_g.new_vertex_property('bool')
    optin.fa = g.vp.optin.fa
    new_g.vertex_properties['optin'] = optin

    ew = new_g.new_edge_property('int')
    new_g.edge_properties['ew'] = ew
    
    mask_in_in = g.edges_in_in() 
    edges_in_in = g.get_edges([g.ep.ew])[mask_in_in]
    new_g.add_edge_list(edges_in_in, eprops=[new_g.ep.ew])

    mask_not_in_in = g.edges_without_in_in()
    edges_not_in_in = g.get_edges()[mask_not_in_in]
    non_zero_edges_weighted = np.concatenate((edges_not_in_in, np.array([ edges_w_noisy_non_zero ]).T ), axis=1)
    edges_non_zero_filt =  non_zero_edges_weighted[non_zero_edges_weighted[:,2] != 0 ]  # filter edges_w != 0
    new_g.add_edge_list(edges_non_zero_filt, eprops=[new_g.ep.ew])

    non_optins_pos = new_g.vp.optin.fa == 0

    num_zero_edges_remaining = len(edges_w_noisy_zeros)
    
    # zero_edges = np.array([])
    zero_edges = np.empty((0,2), int)

    while True: 
        new_edge_pos_1 = np.random.choice(new_g.n(), num_zero_edges_remaining, replace=True)
        new_edge_pos_2 = np.random.choice(new_g.n(), num_zero_edges_remaining, replace=True)
 
        # selecting optin-optin edges
        non_optins_pos_1 = non_optins_pos[new_edge_pos_1]
        non_optins_pos_2 = non_optins_pos[new_edge_pos_2]
        non_optins_zero_edges_pos = non_optins_pos_1 + non_optins_pos_2
        
        # removing optin-optin edges
        new_zero_edges = np.concatenate(( np.array([new_edge_pos_1[non_optins_zero_edges_pos]]), np.array([new_edge_pos_2[non_optins_zero_edges_pos]])), axis=0).T  
        
        new_zero_edges.sort()

        # removing duplicated edges
        new_zero_edges_unique = np.unique(new_zero_edges, axis=0)

        # removing self edges
        new_zero_edges_without_self_edges = new_zero_edges_unique[new_zero_edges_unique[:,0] != new_zero_edges_unique[:,1]]
        
        # removing edges already picked 
        new_zero_edges_not_picked = set(map(tuple, new_zero_edges_without_self_edges)).difference(set(map(tuple, zero_edges)))
       
        # removing edges that already exists 
        new_zero_edges_not_repeated = new_zero_edges_not_picked.difference(set(map(tuple, edges_not_in_in)))

        new_zero_edges_not_repeated_arr = np.array(list(new_zero_edges_not_repeated))

        if len(new_zero_edges_not_repeated_arr) > 0:
            zero_edges = np.append( zero_edges, new_zero_edges_not_repeated_arr , axis=0 )

            num_zero_edges_remaining = len(edges_w_noisy_zeros) - len(zero_edges)
            if num_zero_edges_remaining == 0:
                break

    zero_edges_weighted = np.concatenate((zero_edges, np.array([ edges_w_noisy_zeros ]).T ), axis=1)
    zero_edges_filtered = zero_edges_weighted[zero_edges_weighted[:,2] != 0 ]  # filter edges_w != 0
    new_g.add_edge_list(zero_edges_filtered, eprops=[new_g.ep.ew])

    # verifying consistence in sampling graph
    if new_g.m() != ( len( edges_w_noisy_non_zero[np.nonzero(edges_w_noisy_non_zero)] ) + len( edges_w_noisy_zeros[np.nonzero(edges_w_noisy_zeros)]) + len(edges_in_in) ):
        utils.log_msg('problem in sampling graph')

    return new_g

def get_edges_from_degree_sequence(g, ds):

    non_optins_pos = g.vp.optin.fa == 0
    remaining_edges = int(np.sum(ds)/2)
    existing_edges = g.get_edges()
    len_existing_edges = len(existing_edges)

    probs = ds/np.sum(ds)

    i = 20    # number of attempts the algorithm can run. if it can not find a solution, ends the while
    edges_not_allocated = 0
    while remaining_edges > 0: 
        if i > 0:
            i -= 1
            node_1, node_2 = np.random.choice(g.n(), 2, replace=False, p=probs)
        else: 
            node_1, node_2 = np.random.choice(g.n(), 2, replace=False)
 
        new_edge = np.array([node_1, node_2])
        new_edge.sort()
   
        # check if already exists
        if tuple(new_edge) not in set(map(tuple, existing_edges)):

            non_optins_pos_1 = non_optins_pos[node_1]
            non_optins_pos_2 = non_optins_pos[node_2]

            # check if both are not opt-in
            if non_optins_pos_1 or non_optins_pos_2:
                existing_edges = np.append(existing_edges, np.array([new_edge]), axis=0) 
                remaining_edges -= 1
                if i > 0:
                    ds[node_1] -= 1
                    ds[node_2] -= 1
                    probs = ds/np.sum(ds)
                    i = 20
                else: 
                    edges_not_allocated += 1

    utils.log_msg('random edges created in degree sequence algorithm: %s' % str(edges_not_allocated))
    return existing_edges[len_existing_edges:]        

def get_edges_from_degree_sequence2(g, degree_seq):

    ds = degree_seq.copy()

    non_optins_pos = g.vp.optin.fa == 0
    remaining_edges = np.sum(ds)/2
    # existing_edges = g.get_edges()
    existing_edges = set(map(tuple, g.get_edges()))  

    random_edges_to_be_created = 0
    new_edges = np.empty((0,2), int)

    while remaining_edges > 0: 
        node_with_highest_degree_pos = np.argmax(ds)
        degree_of_highest = ds[node_with_highest_degree_pos]

        ds_probs = ds.copy()
        ds_probs[node_with_highest_degree_pos] = 0

        # if already exists
        ds_already_existing = np.array(sum([x for x in existing_edges if node_with_highest_degree_pos in x], ()))
        ds_already_existing = ds_already_existing[ds_already_existing != node_with_highest_degree_pos]
        if len(ds_already_existing) > 0:
            ds_probs[ds_already_existing] = 0

        # if opt-in
        if ~non_optins_pos[node_with_highest_degree_pos]:
            ds_probs[~non_optins_pos] = 0

        len_ds_probs_still_highers_than_0 = np.sum(ds_probs > 0 )
        if len_ds_probs_still_highers_than_0 >= degree_of_highest:
            nodes_to_create_edge = ds_probs.argsort()[-degree_of_highest:][::-1]
            random_edges_to_be_created_node_x = 0
        else: 
            if len_ds_probs_still_highers_than_0 > 0:
                nodes_to_create_edge = ds_probs.argsort()[-len_ds_probs_still_highers_than_0:][::-1]
            random_edges_to_be_created_node_x = (degree_of_highest - len_ds_probs_still_highers_than_0)

        num_created_edges = min(len_ds_probs_still_highers_than_0, degree_of_highest)

        ds[node_with_highest_degree_pos] = 0

        if len_ds_probs_still_highers_than_0 > 0:
            ds[nodes_to_create_edge] -= 1

            orig = (np.ones(num_created_edges)*node_with_highest_degree_pos).astype(int)
            edges_to_be_added = np.stack((orig, nodes_to_create_edge ), axis=1)
            edges_to_be_added.sort()

            new_edges = np.append( new_edges, edges_to_be_added , axis=0 )
            existing_edges = set(map(tuple, existing_edges)).union(set(map(tuple, edges_to_be_added)))

        remaining_edges -= (num_created_edges +  random_edges_to_be_created_node_x/2 ) 
        random_edges_to_be_created += random_edges_to_be_created_node_x/2

    utils.log_msg('random edges created in degree sequence algorithm: %s' % str(random_edges_to_be_created))
    
    if random_edges_to_be_created > 0:
        random_edges = sample_random_edges(g.n(), int(random_edges_to_be_created), existing_edges, non_optins_pos)
        new_edges = np.append( new_edges, random_edges , axis=0 )

    return new_edges

def adjust_degree_sequence(g, degree_seq, non_optins_pos):
    ds = degree_seq.copy()

    # m = int(np.sum(degree_seq)/2)

    df_edges = pd.DataFrame(data=g.get_edges([g.ep.ew]), columns=["s", "d", "w"], dtype="int")
    df_edges = df_edges.sort_values(by=['w'], ascending=False)
    existing_edges = df_edges.to_numpy()

    new_edges_set = set(map(tuple, []))  
    new_edges_list = np.empty((0,2), int) 
    weights = []
    skipped_edges = []

    # first step: top-down edges addition

    for i in range(len(existing_edges)):
        edge_i = existing_edges[i]
        orig = edge_i[0]
        dest = edge_i[1]
        if ds[orig] != 0 and ds[dest] != 0:
            new_edge = np.array([orig, dest])
            new_edge.sort()
            new_edges_set.add((new_edge[0],new_edge[1]))
            new_edges_list = np.append( new_edges_list, np.array([new_edge]) , axis=0 )
            ds[new_edge[0]] -= 1 
            ds[new_edge[1]] -= 1 
            weights.append(edge_i[2]) 
        else:    
            skipped_edges.append(i)
    
    highest_edges_with_w = np.concatenate((new_edges_list, np.array([ weights ]).T ), axis=1)

    edges_created_based_on_deg_seq = sample_random_edges_based_on_deg_seq(g.n(), len(skipped_edges), new_edges_set, non_optins_pos, ds )
    all_edges_so_far = new_edges_set.union( set(map(tuple, edges_created_based_on_deg_seq)) )

    num_remaining_edges = len(skipped_edges) - len(edges_created_based_on_deg_seq) 

    utils.log_msg("num random edges added: %s" % num_remaining_edges)

    if num_remaining_edges > 0:
        edges_created_randomly = sample_random_edges(g.n(), num_remaining_edges, all_edges_so_far, non_optins_pos)
    else:
        edges_created_randomly = np.empty((0,2), int)

    edges_created_after_first_addition = np.append( edges_created_based_on_deg_seq, edges_created_randomly, axis=0 ) 
    existing_edges_after_first_addition = existing_edges[skipped_edges]
    new_random_edges_with_w = np.concatenate((edges_created_after_first_addition, np.array([ existing_edges_after_first_addition[:,2] ]).T ), axis=1)

    edges_with_degree_adjusted = np.concatenate((highest_edges_with_w, new_random_edges_with_w), axis=0)
    return edges_with_degree_adjusted

def sample_random_edges_based_on_deg_seq(n, num_edges_to_be_sampled, existing_edges, non_optins_pos, deg_seq):

    num_edges_remaining = num_edges_to_be_sampled
    picked_edges = np.empty((0,2), int)
    num_attempts_until_break = 0

    while True: 
        positive_deg_seq_pos = np.where(deg_seq != 0)[0] 
        remaining_deg_seq = np.zeros(n, dtype=int)
        remaining_deg_seq[positive_deg_seq_pos] = 1
        probs = remaining_deg_seq/np.sum(remaining_deg_seq)
        new_edge_pos_1 = np.random.choice(n, num_edges_remaining, replace=True, p=probs)
        new_edge_pos_2 = np.random.choice(n, num_edges_remaining, replace=True, p=probs)
 
        # selecting optin-optin edges
        non_optins_pos_1 = non_optins_pos[new_edge_pos_1]
        non_optins_pos_2 = non_optins_pos[new_edge_pos_2]
        non_optins_zero_edges_pos = non_optins_pos_1 + non_optins_pos_2
        
        # removing optin-optin edges
        new_zero_edges = np.concatenate(( np.array([new_edge_pos_1[non_optins_zero_edges_pos]]), np.array([new_edge_pos_2[non_optins_zero_edges_pos]])), axis=0).T  
        
        new_zero_edges.sort()

        # removing duplicated edges
        new_zero_edges_unique = np.unique(new_zero_edges, axis=0)

        # removing self edges
        new_zero_edges_without_self_edges = new_zero_edges_unique[new_zero_edges_unique[:,0] != new_zero_edges_unique[:,1]]
        
        # removing edges already picked 
        new_zero_edges_not_picked = set(map(tuple, new_zero_edges_without_self_edges)).difference(set(map(tuple, picked_edges)))
       
        # removing edges that already exists 
        new_zero_edges_not_repeated = new_zero_edges_not_picked.difference(existing_edges)

        new_zero_edges_not_repeated_arr = np.array(list(new_zero_edges_not_repeated))

        for e in new_zero_edges_not_repeated_arr:
            if deg_seq[e[0]] != 0 and deg_seq[e[1]] != 0:
                picked_edges = np.append( picked_edges, [e] , axis=0 )
                deg_seq[e[0]] -= 1
                deg_seq[e[1]] -= 1

        # if len(new_zero_edges_not_repeated_arr) > 0:
        #     picked_edges = np.append( picked_edges, new_zero_edges_not_repeated_arr , axis=0 )

        last_num_edges_remaining = num_edges_remaining
        num_edges_remaining = num_edges_to_be_sampled - len(picked_edges)
        if last_num_edges_remaining == num_edges_remaining:
            num_attempts_until_break += 1
            if num_attempts_until_break == 5: 
                break
        if num_edges_remaining == 0:
            break
    
    return picked_edges

def adjust_degree_sequence2(g, degree_seq, non_optins_pos):
    ds = degree_seq.copy()

    # m = int(np.sum(degree_seq)/2)

    df_edges = pd.DataFrame(data=g.get_edges([g.ep.ew]), columns=["s", "d", "w"], dtype="int")
    df_edges = df_edges.sort_values(by=['w'], ascending=False).reset_index(drop=True)
    existing_edges = df_edges.to_numpy()

    # new_edges = set(map(tuple, []))  
    new_edges_list = np.empty((0,2), int) 
    weights = np.array([])
    existing_edges_to_keep = []

    idx = np.array([], int)
    for i in range(len(ds)):
        idx = np.append(idx, df_edges[(df_edges['s'] == i) | (df_edges['d'] == i)].head(ds[i]).index)
    
    u, c = np.unique(idx, return_counts=True)
    idx_to_keep = u[c > 1]

    idx_to_keep_mask = np.zeros(g.m(), dtype=bool)
    idx_to_keep_mask[idx_to_keep] = True 

    new_edges_list = existing_edges[idx_to_keep_mask][:,(0,1)]
    new_edges = set(map(tuple, new_edges_list))
    weights = existing_edges[idx_to_keep_mask][:,2]

    # existing_edges_to_keep = existing_edges[~idx_to_keep_mask][:,(0,1)]
    existing_edges_to_keep = np.where(~idx_to_keep_mask)[0]

    all_ids = new_edges_list.flatten()
    unique, counts = np.unique(all_ids, return_counts=True)

    for i in range(len(ds)):
        ds[unique[i]] -= counts[i]

    # first step: top-down edges addition
    # for i in range(len(existing_edges)):
    #     edge_i = existing_edges[i]
    #     orig = edge_i[0]
    #     dest = edge_i[1]
    #     if ds[orig] != 0 and ds[dest] != 0:
    #         new_edge = np.array([orig, dest])
    #         new_edge.sort()
    #         new_edges.add((new_edge[0],new_edge[1]))
    #         new_edges_list = np.append( new_edges_list, np.array([new_edge]) , axis=0 )
    #         ds[new_edge[0]] -= 1
    #         ds[new_edge[1]] -= 1
    #         weights.append(edge_i[2])
    #     else:    
    #         existing_edges_to_keep.append(i)
    
    if (np.sum(ds) % 2) != 0:
        ds_positives = np.where(ds > 0)[0]
        id_to_decrease_1_unit = np.random.choice(ds_positives, 1)
        ds[id_to_decrease_1_unit] -= 1
    random_g = gt.generation.random_graph(len(ds), lambda i: ds[i], directed=False)
    random_edges = random_g.get_edges()
    
    # random_edges_sorted = []
    # for e in random_edges:
    #     random_edges_sorted.append(sorted(e))

    df_edges = pd.DataFrame(data=random_edges.astype('int'), columns=["s", "d"])
    df_edges['or'] = df_edges[['s','d']].min(axis=1)
    df_edges['de'] = df_edges[['s','d']].max(axis=1)
    df_edges = df_edges.sort_values(by=['or', 'de'])
    df_edges = df_edges[['or','de']]
    random_edges_sorted = df_edges.to_numpy()

    random_edges_set = set(map(tuple, random_edges_sorted))
    intersected_edges = random_edges_set.intersection(new_edges)
    num_intersecetd_edges = len(intersected_edges)
    utils.log_msg("num random edges added: %s" % num_intersecetd_edges)
    random_edges_set_difference = random_edges_set.difference(intersected_edges)
    random_edges_arr_difference = np.array([list(item) for item in random_edges_set_difference]) # np.append( new_edges_list, np.array([new_edge]) , axis=0 )
    
    all_edges_so_far = new_edges.union(random_edges_set_difference)

    if num_intersecetd_edges > 0:
        new_random_edges = sample_random_edges(g.n(), num_intersecetd_edges, all_edges_so_far, non_optins_pos)
    else:
        new_random_edges = np.empty((0,2), int)

    # while num_intersecetd_edges > 0:
    #     u, v = np.random.choice(g.n(), 2, replace=False)
    #     new_e = np.array([u, v])
    #     new_e.sort()
   
    #     # check if not exists
    #     if tuple(new_e) not in set(map(tuple, all_edges_so_far)):
    #         all_edges_so_far.add(tuple(new_e))
    #         random_edges_arr_difference = np.append(random_edges_arr_difference, np.array([new_e]) , axis=0 )
    #         num_intersecetd_edges -= 1

    random_edges_arr_difference = np.append(random_edges_arr_difference, new_random_edges , axis=0 )
    existing_edges_after_first_addition = existing_edges[existing_edges_to_keep]

    highest_edges_with_w = np.concatenate((new_edges_list, np.array([ weights ]).T ), axis=1)
    new_random_edges_with_w = np.concatenate((random_edges_arr_difference, np.array([ existing_edges_after_first_addition[:,2] ]).T ), axis=1)

    concatened_edges = np.concatenate((highest_edges_with_w, new_random_edges_with_w), axis=0)
    return concatened_edges


def adjust_degree_sequence_old(g, degree_seq):
    ds = degree_seq.copy()
    non_optins_pos = g.vp.optin.fa == 0

    m = int(np.sum(degree_seq)/2)

    df_edges = pd.DataFrame(data=g.get_edges([g.ep.ew]), columns=["s", "d", "w"], dtype="int")
    df_edges = df_edges.sort_values(by=['w'], ascending=False)
    existing_edges = df_edges.to_numpy()

    new_edges = set(map(tuple, []))  
    new_edges_list = np.empty((0,2), int) 
    weights = []
    existing_edges_to_keep = []

    # first step: top-down edges addition
    for i in range(len(existing_edges)):
        edge_i = existing_edges[i]
        orig = edge_i[0]
        dest = edge_i[1]
        if ds[orig] != 0 and ds[dest] != 0:
            new_edge = np.array([orig, dest])
            new_edge.sort()
            new_edges.add((new_edge[0],new_edge[1]))
            new_edges_list = np.append( new_edges_list, np.array([new_edge]) , axis=0 )
            ds[new_edge[0]] -= 1
            ds[new_edge[1]] -= 1
            weights.append(edge_i[2])
        else:    
            existing_edges_to_keep.append(i)
            
    # second step: one side is ok

    existing_edges_2 = existing_edges[existing_edges_to_keep]
    existing_edges_to_keep = []

    for i in range(len(existing_edges_2)):
        edge_i = existing_edges_2[i]
        orig = edge_i[0]
        dest = edge_i[1]
        if ds[orig] != 0 or ds[dest] != 0:
            if ds[orig] != 0:
                v1 = orig
            else:
                v1 = dest

            highest_dss = ds.argsort()[-(len(ds)):][::-1]
            for j in range(len(highest_dss)):

                v2 = highest_dss[j]

                if non_optins_pos[v1] or non_optins_pos[v2]:
                    new_edge = np.array([v1, v2])
                    new_edge.sort()

                    if new_edge[0] != new_edge[1]:
                        if tuple(new_edge) not in set(map(tuple, new_edges)):
                            new_edges.add((new_edge[0],new_edge[1]))
                            new_edges_list = np.append( new_edges_list, np.array([new_edge]) , axis=0 )
                            ds[new_edge[0]] -= 1
                            ds[new_edge[1]] -= 1
                            weights.append(edge_i[2])
                            break 
                
                if j == len(highest_dss)-1:
                    existing_edges_to_keep.append(i)
        else:
            existing_edges_to_keep.append(i)

    # third step: graph realization

    existing_edges_3 = existing_edges_2[existing_edges_to_keep]

    for i in range(len(existing_edges_3)):
        highest_dss = ds.argsort()[-(len(ds)):][::-1]
        for a, b in itertools.combinations( list(range( len(highest_dss) )) , 2):
            if non_optins_pos[ highest_dss[a]  ] or non_optins_pos[ highest_dss[b] ]:
                new_edge = np.array([highest_dss[a], highest_dss[b]])
                new_edge.sort()

                if new_edge[0] != new_edge[1]:
                    if tuple(new_edge) not in set(map(tuple, new_edges)):
                        new_edges.add((new_edge[0],new_edge[1]))
                        new_edges_list = np.append( new_edges_list, np.array([new_edge]) , axis=0 )
                        ds[new_edge[0]] -= 1
                        ds[new_edge[1]] -= 1
                        weights.append(edge_i[2])
                        break
    
    new_edges_with_w = np.concatenate((new_edges_list, np.array([ weights ]).T ), axis=1)

    return new_edges_with_w
           
        
def remove_edges_with_lower_weights(g, m):

    existing_edges = g.get_edges([g.ep.ew])
    exceeding_edges = len(existing_edges) - m

    if exceeding_edges > 0:

        df_edges = pd.DataFrame(data=existing_edges, columns=["s", "d", "w"], dtype="int")
        df_edges = df_edges.sort_values(by=['w'], ascending=False)
        df_edges = df_edges[:m]
        new_edges = df_edges.to_numpy()

        # edges_w = new_edges[:,2]
        # edges_w_adjusted = min_l2_norm(edges_w, sum_w, num_steps=10)

        # new_edges_out_out = np.concatenate((new_edges[:,[0,1]], np.array([edges_w_adjusted]).T ), axis=1)
        # edges_in_out = g.get_edges([g.ep.ew])[g.edges_in_out()]
        # edges_to_be_added = np.append(edges_in_out, new_edges_out_out, axis=0)  

    else:
        non_optins_positions = g.vp.optin.fa == 0
        edges_to_be_added = sample_random_edges(g.n(), np.absolute(exceeding_edges), existing_edges[:,[0,1]], non_optins_positions)
        edges_to_be_added_with_w = np.concatenate((edges_to_be_added, np.array([np.ones( len(edges_to_be_added) )]).T ), axis=1)

        new_edges = np.append(existing_edges, edges_to_be_added_with_w, axis=0)  

    return new_edges

def top_m_edges_with_lower_weights(edges, m, g_basis):

    exceeding_edges = len(edges) - m
    if exceeding_edges > 0:

        df_edges = pd.DataFrame(data=edges.astype('int'), columns=["s", "d", "w"])
        df_edges = df_edges.sort_values(by=['w'], ascending=False)
        df_edges = df_edges[:m]
        new_edges = df_edges.to_numpy()

    else:
        non_optins_positions = g_basis.vp.optin.fa == 0
        edges_to_be_added = sample_random_edges(g_basis.n(), np.absolute(exceeding_edges), edges[:,[0,1]], non_optins_positions)
        edges_to_be_added_with_w = np.concatenate((edges_to_be_added, np.array([np.ones( len(edges_to_be_added) )]).T ), axis=1)

        new_edges = np.append(edges, edges_to_be_added_with_w, axis=0)  

    return new_edges

def filter_edges_higher_than_threshold(g, threshold):
    edges = g.get_edges([g.ep.ew])
    edges_w = edges[:,2].astype('int')
    edges_filtered_pos = np.where(edges_w >= threshold)[0]
    new_edges = edges[edges_filtered_pos]
    return new_edges

def sample_random_edges(n, num_edges_to_be_sampled, existing_edges, non_optins_pos):

    num_edges_remaining = num_edges_to_be_sampled
    picked_edges = np.empty((0,2), int)

    while True: 
        new_edge_pos_1 = np.random.choice(n, num_edges_remaining, replace=True)
        new_edge_pos_2 = np.random.choice(n, num_edges_remaining, replace=True)
 
        # selecting optin-optin edges
        non_optins_pos_1 = non_optins_pos[new_edge_pos_1]
        non_optins_pos_2 = non_optins_pos[new_edge_pos_2]
        non_optins_zero_edges_pos = non_optins_pos_1 + non_optins_pos_2
        
        # removing optin-optin edges
        new_zero_edges = np.concatenate(( np.array([new_edge_pos_1[non_optins_zero_edges_pos]]), np.array([new_edge_pos_2[non_optins_zero_edges_pos]])), axis=0).T  
        
        new_zero_edges.sort()

        # removing duplicated edges
        new_zero_edges_unique = np.unique(new_zero_edges, axis=0)

        # removing self edges
        new_zero_edges_without_self_edges = new_zero_edges_unique[new_zero_edges_unique[:,0] != new_zero_edges_unique[:,1]]
        
        # removing edges already picked 
        new_zero_edges_not_picked = set(map(tuple, new_zero_edges_without_self_edges)).difference(set(map(tuple, picked_edges)))
       
        # removing edges that already exists 
        new_zero_edges_not_repeated = new_zero_edges_not_picked.difference(existing_edges)

        new_zero_edges_not_repeated_arr = np.array(list(new_zero_edges_not_repeated))

        if len(new_zero_edges_not_repeated_arr) > 0:
            picked_edges = np.append( picked_edges, new_zero_edges_not_repeated_arr , axis=0 )

            num_edges_remaining = num_edges_to_be_sampled - len(picked_edges)
            if num_edges_remaining == 0:
                break
    
    return picked_edges

def mean_of_duplicated_edges(edges):
    df_edges = pd.DataFrame(data=edges, columns=["s", "d", "w"], dtype="int")

    df_edges['or'] = df_edges[['s','d']].min(axis=1)
    df_edges['de'] = df_edges[['s','d']].max(axis=1)

    df_edges = df_edges.sort_values(by=['or', 'de'])
    df_edges = df_edges[['or','de','w']]
    df_edges = df_edges.groupby(by=['or','de'], as_index=False).mean()
    
    df_edges = df_edges.round().astype(int)
    new_edges = df_edges.to_numpy()

    return new_edges

def mean_of_all_edges(edges):
    df_edges = pd.DataFrame(data=edges, columns=["s", "d", "w"], dtype="int")

    df_edges['or'] = df_edges[['s','d']].min(axis=1)
    df_edges['de'] = df_edges[['s','d']].max(axis=1)

    df_edges = df_edges.sort_values(by=['or', 'de'])
    df_edges = df_edges[['or','de','w']]
    df_edges = df_edges.groupby(by=['or','de'], as_index=False).mean()
    # df_edges['w'] = (df_edges['w']/2).apply(np.ceil).astype(int)
    
    df_edges = df_edges.round().astype(int)
    new_edges = df_edges.to_numpy()

    return new_edges

def build_g_from_edges(g, new_edges, allow_zero_edges_w=False, add_optin_edges=True):

    new_g = WGraph()
    new_g.add_vertex(n=g.n())
    optin = new_g.new_vertex_property('bool')
    optin.fa = g.vp.optin.fa
    new_g.vertex_properties['optin'] = optin

    ew = new_g.new_edge_property('int')
    new_g.edge_properties['ew'] = ew

    if add_optin_edges:
        mask_in_in = g.edges_in_in() 
        edges_in_in = g.get_edges([g.ep.ew])[mask_in_in]
    else: 
        edges_in_in = np.empty((0,3), int)
    # new_g.add_edge_list(edges_in_in, eprops=[new_g.ep.ew])

    # selection of only edges with weigth != 0
    if not allow_zero_edges_w:
        edges_w = new_edges[:,2]
        non_zero_edges_w_mask = edges_w != 0
        edges_to_be_added = np.append(edges_in_in, new_edges[non_zero_edges_w_mask], axis=0) 
    else:
        edges_to_be_added = np.append(edges_in_in, new_edges, axis=0) 

    df_edges = pd.DataFrame(data=edges_to_be_added.astype('int'), columns=["s", "d", "w"])
    df_edges['or'] = df_edges[['s','d']].min(axis=1)
    df_edges['de'] = df_edges[['s','d']].max(axis=1)
    df_edges = df_edges.sort_values(by=['or', 'de'])
    df_edges = df_edges[['or','de','w']]
    edges_to_be_added_sorted = df_edges.to_numpy()

    new_g.add_edge_list(edges_to_be_added_sorted, eprops=[new_g.ep.ew])

    # if new_g.m() != ( len( edges_to_be_added) + len(edges_in_in) ):
    #     utils.error_msg('problem in sampling graph')

    return new_g   

def remove_edges_with_lower_weights_and_adjust(g, m, sum_w):
    edges_out_out = g.get_edges([g.ep.ew])[g.edges_out_out()]

    exceeding_edges = len(edges_out_out) - m
    if exceeding_edges > 0:

        df_edges = pd.DataFrame(data=edges_out_out, columns=["s", "d", "w"], dtype="int")
        df_edges = df_edges.sort_values(by=['w'], ascending=False)
        df_edges = df_edges[:m]
        new_edges = df_edges.to_numpy()

        edges_w = new_edges[:,2]
        edges_w_adjusted = min_l2_norm_old(edges_w, sum_w, num_steps=10)

        new_edges_out_out = np.concatenate((new_edges[:,[0,1]], np.array([edges_w_adjusted]).T ), axis=1)
        edges_in_out = g.get_edges([g.ep.ew])[g.edges_in_out()]
        edges_to_be_added = np.append(edges_in_out, new_edges_out_out, axis=0)  
        new_g = build_g_from_edges(g, edges_to_be_added)

        return new_g

    else:
        return g

def adjust_edge_weights_based_on_ns_old(g, node_strengths):

    nss = node_strengths.copy()

    mask_out_out = g.edges_out_out()
    g_out_out = WGraph(G=gt.GraphView(g, efilt=mask_out_out), prune=True)

    av = g.new_edge_property('bool') # already visited
    g.edge_properties['av'] = av

    optins = g.optins()
    optouts = g.optouts()
    # nss = g.node_strengths()

    new_edges = np.empty((0,3), int)

    for v in optins:
        neighbors_v = g.get_out_edges(v, [g.ep.ew, g.edge_index] )
        if len(neighbors_v) > 0:
            edges_w = neighbors_v[:,2] 
            edges_w_adjusted = min_l2_norm_old(edges_w, nss[v], num_steps=10)
            edges_adjusted = np.concatenate((neighbors_v[:,[0,1]], np.array([edges_w_adjusted]).T ), axis=1)
            new_edges = np.append(new_edges, edges_adjusted, axis=0)  
            nss[neighbors_v[:,1]] = nss[neighbors_v[:,1]] - edges_w_adjusted
            g.ep.av.fa[neighbors_v[:,3]] = 1

    degrees_out_out = g_out_out.degrees()[optouts]
    optouts_sorted = np.argsort(degrees_out_out)

    for o in optouts_sorted:
        v = optouts[o]
        neighbors_v_not_filtered = g.get_out_edges(v, [g.ep.ew, g.ep.av, g.edge_index] )
        neighbors_v = neighbors_v_not_filtered[neighbors_v_not_filtered[:,3] == 0]
        if len(neighbors_v) > 0:
            edges_w = neighbors_v[:,2] 
            edges_w_adjusted = min_l2_norm_old(edges_w, nss[v], num_steps=10)
            edges_adjusted = np.concatenate((neighbors_v[:,[0,1]], np.array([edges_w_adjusted]).T ), axis=1)
            new_edges = np.append(new_edges, edges_adjusted, axis=0)  
            nss[neighbors_v[:,1]] = nss[neighbors_v[:,1]] - edges_w_adjusted
            g.ep.av.fa[neighbors_v[:,4]] = 1

    return new_edges

def adjust_edge_weights_based_on_ns(g, node_strengths):

    nss = g.node_strengths().copy()
    nss_desired = node_strengths.copy()

    mask_out_out = g.edges_out_out()
    g_out_out = WGraph(G=gt.GraphView(g, efilt=mask_out_out), prune=True)

    av = g.new_edge_property('bool') # already visited
    g.edge_properties['av'] = av

    optins = g.optins()
    optouts = g.optouts()

    new_edges = np.empty((0,3), int)

    # for v in optins:
    #     neighbors_v = g.get_out_edges(v, [g.ep.ew, g.edge_index] )
    #     if len(neighbors_v) > 0:
    #         edges_w = neighbors_v[:,2] 
    #         edges_w_adjusted = min_l2_norm_old(edges_w, nss[v], num_steps=10)
    #         edges_adjusted = np.concatenate((neighbors_v[:,[0,1]], np.array([edges_w_adjusted]).T ), axis=1)
    #         new_edges = np.append(new_edges, edges_adjusted, axis=0)  
    #         nss[neighbors_v[:,1]] = nss[neighbors_v[:,1]] - edges_w_adjusted
    #         g.ep.av.fa[neighbors_v[:,3]] = 1

    # degrees_out_out = g_out_out.degrees()[optouts]
    # optouts_sorted = np.argsort(degrees_out_out)

    nodes_not_visited = np.ones( g.n() , dtype=bool)
    highest_difference_idx = 1
    nss_difference = np.abs( nss - nss_desired ) 

    while np.sum(nodes_not_visited) != 0: 

        v = np.argsort(nss_difference)[-highest_difference_idx]

        if nodes_not_visited[v]:

    # for o in optouts_sorted:
    #     v = optouts[o]
            neighbors_v_not_filtered = g.get_out_edges(v, [g.ep.ew, g.ep.av, g.edge_index] )
            neighbors_v = neighbors_v_not_filtered[neighbors_v_not_filtered[:,3] == 0]
            if len(neighbors_v) > 0:
                edges_w = neighbors_v[:,2] 
                edges_w_adjusted = min_l2_norm_old(edges_w, nss_desired[v], num_steps=10)
                edges_adjusted = np.concatenate((neighbors_v[:,[0,1]], np.array([edges_w_adjusted]).T ), axis=1)
                new_edges = np.append(new_edges, edges_adjusted, axis=0)  
                difference_of_w = edges_w - edges_w_adjusted
                nss[neighbors_v[:,1]] = nss[neighbors_v[:,1]] - difference_of_w
                nss[v] = nss[v] - np.sum(difference_of_w)
                g.ep.av.fa[neighbors_v[:,4]] = 1
                highest_difference_idx = 1
                nss_difference = np.abs( (nss - nss_desired)/nss_desired ) 
            nodes_not_visited[v] = False
        else: 
            highest_difference_idx += 1

    return new_edges

def adjust_edge_weights_based_on_ns3(g, node_strengths):

    nss = g.node_strengths().copy()
    nss_desired = node_strengths.copy()

    mask_out_out = g.edges_out_out()
    g_out_out = WGraph(G=gt.GraphView(g, efilt=mask_out_out), prune=True)

    av = g.new_edge_property('bool') # already visited
    g.edge_properties['av'] = av

    optins = g.optins()
    optouts = g.optouts()

    new_edges = np.empty((0,3), int) 

    # for v in optins:
    #     neighbors_v = g.get_out_edges(v, [g.ep.ew, g.edge_index] )
    #     if len(neighbors_v) > 0:
    #         edges_w = neighbors_v[:,2] 
    #         edges_w_adjusted = min_l2_norm_old(edges_w, nss[v], num_steps=10)
    #         edges_adjusted = np.concatenate((neighbors_v[:,[0,1]], np.array([edges_w_adjusted]).T ), axis=1)
    #         new_edges = np.append(new_edges, edges_adjusted, axis=0)  
    #         nss[neighbors_v[:,1]] = nss[neighbors_v[:,1]] - edges_w_adjusted
    #         g.ep.av.fa[neighbors_v[:,3]] = 1

    # degrees_out_out = g_out_out.degrees()[optouts]
    # optouts_sorted = np.argsort(degrees_out_out)

    nodes_not_visited = np.ones( g.n() , dtype=bool)
    highest_difference_idx = 1
    nss_difference = np.abs( nss - nss_desired ) 

    while np.sum(nodes_not_visited) != 0: 

        v = np.argsort(nss_difference)[-highest_difference_idx]

        if nodes_not_visited[v]:

    # for o in optouts_sorted:
    #     v = optouts[o]
            neighbors_v_not_filtered = g.get_out_edges(v, [g.ep.ew, g.ep.av, g.edge_index] )
            neighbors_v = neighbors_v_not_filtered[neighbors_v_not_filtered[:,3] == 0]
            if len(neighbors_v) > 0:
                edges_w = neighbors_v[:,2] 
                edges_w_adjusted = min_l2_norm_old(edges_w, nss_desired[v], num_steps=10)
                edges_adjusted = np.concatenate((neighbors_v[:,[0,1]], np.array([edges_w_adjusted]).T ), axis=1)
                new_edges = np.append(new_edges, edges_adjusted, axis=0)  
                difference_of_w = edges_w - edges_w_adjusted
                nss[neighbors_v[:,1]] = nss[neighbors_v[:,1]] - difference_of_w
                nss[v] = nss[v] - np.sum(difference_of_w)
                g.ep.av.fa[neighbors_v[:,4]] = 1
                highest_difference_idx = 1
                nss_difference = np.abs( (nss - nss_desired)/nss_desired ) 
            nodes_not_visited[v] = False
        else: 
            highest_difference_idx += 1

    return new_edges

# def adjust_edge_weights_based_on_ns(g, ns):
    
#     n = g.n()
#     upper_triangle_idx = np.triu_indices(n, k=1)
#     edges_w_matrix = sp.adjacency(g, weight=g.ep.ew).toarray()

#     # edges_w_triu = np.zeros([n,n])
#     # for i in range(n):
#     #     for j in range(n):
#     #         if i < j:
#     #             edges_w_triu[i][j] = edges_w_matrix[i][j]

#     x = cp.Variable(( n, n ))
#     objective = cp.Minimize(cp.sum_squares( x - edges_w_matrix ))

#     constraints_string = '[ x >= 0, x == x.T, ' 
#     for i in range(n):
#         constraints_string += 'sum(x[%s]) == ns[%s] ,' % (i,i)
#         constraints_string += 'x[%s][%s] == 0 ,' % (i,i)
#     constraints_string = constraints_string[:-1]
#     constraints_string += ' ]'
#     constraints = eval(constraints_string)

#     prob = cp.Problem(objective, constraints)
#     prob.solve()

#     new_edges_w_rounded = np.abs(np.around(x.value))
#     new_edges_w = np.clip(new_edges_w_rounded, 0, None)

#     new_edges_w_arr = new_edges_w[upper_triangle_idx]

#     print(1)

