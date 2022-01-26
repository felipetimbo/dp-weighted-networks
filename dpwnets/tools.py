import math
import pandas as pd
import numpy as np
import graph_tool as gt
import cvxpy as cp
import graph_tool.spectral as sp
import os
import itertools

import matplotlib
import matplotlib.pyplot as plt


from dpwnets import (utils, graphics)
from graph.wgraph import WGraph
from graph_tool.util import find_edge

np.random.seed(0)

kwargs = {'linewidth' : 3.5}
font = {'weight' : 'normal', 'size'   : 24}
matplotlib.rc('font', **font)

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

def gradient_descent(init, steps, grad, laplaced_sum, proj=lambda x: x, min_value=1):
    xs = [init]
    for step in steps:
        xs_next = proj(xs[-1] - step * grad(xs[-1]), laplaced_sum, min_value)
        xs = [xs_next]
    return xs

def gradient_descent2(init, steps, grad, proj):
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

def min_l2_norm_old(edges_w, desired_sum, num_steps=500, min_value=1):
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
    plt.plot(range(len(ys)), ys, **kwargs)
    path = "./a.png"
    dir_path = os.path.dirname(os.path.realpath(path))
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(path, dpi=900)

def min_l2_norm2(edges, nss, weight=1, num_steps=500, min_value=1):
    
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

    alpha = 10.0
    
    objective = lambda x: least_squares(A, b, x, num_lines)
    gradient = lambda x: least_squares_gradient(A, b, x, num_lines)
    xs = gradient_descent2(x0, [alpha]*num_steps, gradient, proj2)
    error_plot([objective(x) for x in xs])
    new_edges_w = xs[-1].astype(int)

    return new_edges_w

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

def adjust_degree_sequence(g, degree_seq):
    ds = degree_seq.copy()
    non_optins_pos = g.vp.optin.fa == 0
    # existing_edges = g.get_edges([])  

    m = int(np.sum(degree_seq)/2)

    df_edges = pd.DataFrame(data=g.get_edges([g.ep.ew]), columns=["s", "d", "w"], dtype="int")
    df_edges = df_edges.sort_values(by=['w'], ascending=False)
    existing_edges = df_edges.to_numpy()

    # existing_edges = g.get_edges([g.ep.ew])
    exceeding_edges = len(existing_edges) - m

    if exceeding_edges > 0:

        # random_edges = 0
        # weights_for_random_edges = []
        new_edges = set(map(tuple, []))  
        new_edges_list = np.empty((0,2), int) 
        weights = []

        i = 0
        # while (len(new_edges) + random_edges) <= m:
        while (len(new_edges_list) <= m) and (i < len(existing_edges)) and (np.sum(ds) > 1):
            i += 1
            edge_i = existing_edges[i]
            orig = edge_i[0]
            dest = edge_i[1]
            if ds[orig] != 0 and ds[dest] != 0:
                new_edge = np.array([orig, dest])
                new_edge.sort()
                if new_edge[0] != new_edge[1]:
                    if tuple(new_edge) not in set(map(tuple, new_edges)):
                        new_edges.add((new_edge[0],new_edge[1]))
                        new_edges_list = np.append( new_edges_list, np.array([new_edge]) , axis=0 )
                        ds[new_edge[0]] -= 1
                        ds[new_edge[1]] -= 1
                        weights.append(edge_i[2])
                    # else: 
                    #     print(1) # implementar
            elif ds[orig] != 0 or ds[dest] != 0: # when at least one is not 0
                if ds[orig] != 0:
                    v1 = orig
                else:
                    v1 = dest

                # v1_optout = non_optins_pos[v1]
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

                # random_edges += 1  
                # weights_for_random_edges.append( edge_i[2] )         

            else: # when both are 0
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
                
                # random_edges += 1
                # weights_for_random_edges.append( edge_i[2] )    

        new_edges_with_w = np.concatenate((new_edges_list, np.array([ weights ]).T ), axis=1)

        if len(new_edges_with_w) < m:
            num_random_edges = m - len(new_edges_with_w)
            edges_to_be_added = sample_random_edges(g.n(), num_random_edges, set(map(tuple, new_edges_with_w[:,[0,1]])), non_optins_pos)
            edges_to_be_added_with_w = np.concatenate((edges_to_be_added, np.array([np.ones( len(edges_to_be_added) )]).T ), axis=1)

            new_edges_with_w = np.append(new_edges_with_w, edges_to_be_added_with_w, axis=0) 

        # while True:
        #     v_with_minimum_degree_pos = np.argmin(ds)  # node that has exceding edges 
        #     degree_of_minimum = ds[v_with_minimum_degree_pos ]
        #     if degree_of_minimum >= 0:
        #         break  
        
        # edges_of_v = g.get_out_edges(v_with_minimum_degree_pos, [g.ep.ew] )
        # neighbors = edges_of_v

    else:

        edges_to_be_added = sample_random_edges(g.n(), np.absolute(exceeding_edges), set(map(tuple, existing_edges[:,[0,1]])) , non_optins_pos)
        edges_to_be_added_with_w = np.concatenate((edges_to_be_added, np.array([np.ones( len(edges_to_be_added) )]).T ), axis=1)

        new_edges_with_w = np.append(existing_edges, edges_to_be_added_with_w, axis=0)  

    return new_edges_with_w

    # num_movements_needed = np.sum( np.absolute( degree_seq ))
            
        
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

    df_edges = pd.DataFrame(data=edges_to_be_added, columns=["s", "d", "w"], dtype="int")
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
        edges_w_adjusted = min_l2_norm(edges_w, sum_w, num_steps=10)

        new_edges_out_out = np.concatenate((new_edges[:,[0,1]], np.array([edges_w_adjusted]).T ), axis=1)
        edges_in_out = g.get_edges([g.ep.ew])[g.edges_in_out()]
        edges_to_be_added = np.append(edges_in_out, new_edges_out_out, axis=0)  
        new_g = build_g_from_edges(g, edges_to_be_added)

        return new_g

    else:
        return g


def adjust_edge_weights_based_on_ns(g, nss):

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
            edges_w_adjusted = min_l2_norm(edges_w, nss[v], numbuild_g_from_edges_steps=10)
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
            edges_w_adjusted = min_l2_norm(edges_w, nss[v], num_steps=10)
            edges_adjusted = np.concatenate((neighbors_v[:,[0,1]], np.array([edges_w_adjusted]).T ), axis=1)
            new_edges = np.append(new_edges, edges_adjusted, axis=0)  
            nss[neighbors_v[:,1]] = nss[neighbors_v[:,1]] - edges_w_adjusted
            g.ep.av.fa[neighbors_v[:,4]] = 1

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