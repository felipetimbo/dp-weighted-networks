import math
import numpy as np

from dpwnets import utils
from graph.wgraph import WGraph

def high_pass_filter(arr, eps, max_len):
    a = np.exp(-eps)
    threshold = len(arr) # number of edges in g_without_in_in
    theta = math.log( ((1+a)*threshold)/(max_len - threshold)) / math.log(a)
    
    non_zero_edges_w_filtered_mask = arr >= theta
    non_zero_edges_w_filtered_pos = np.where(non_zero_edges_w_filtered_mask)[0]
    
    num_remaining_edges = np.sum( non_zero_edges_w_filtered_mask )
    
    p = (a**theta)/(1 + a)
    len_binomial = max_len - threshold
    k = np.random.binomial( len_binomial, p)
    
    num_edges_after_filter = num_remaining_edges + k
    num_exceding_edges = int(num_edges_after_filter - threshold)
    
    if num_exceding_edges < 0:
        utils.error_msg('size of high pass filter less than desired')
    
    non_zero_edges_w_filtered = arr[non_zero_edges_w_filtered_mask]
    zeros_edges_w_filtered = np.array(list(map(lambda u: math.log( (1 - u)/a, a) + theta + 1, np.random.uniform(0, 1, k))))
    top_edges_after_filter = np.append(non_zero_edges_w_filtered, zeros_edges_w_filtered)
    
    m_pos = top_edges_after_filter.argsort()[:num_exceding_edges]
    non_zero_edges_w_to_be_removed = m_pos[m_pos < num_remaining_edges]
    if len(non_zero_edges_w_to_be_removed)>0:
        non_zero_edges_w_filtered_mask[non_zero_edges_w_filtered_pos[non_zero_edges_w_to_be_removed]] = False
        utils.log_msg('%s non zero edges removed' % len(non_zero_edges_w_to_be_removed))
        num_remaining_edges -= len(non_zero_edges_w_to_be_removed)
    
    zero_edges_w_to_be_removed = m_pos[m_pos >= num_remaining_edges]    
    utils.log_msg('%s zero edges removed' % len(zero_edges_w_to_be_removed))          
    
    top_edges_after_filter_mask = np.full(len(top_edges_after_filter), True)
    top_edges_after_filter_mask[non_zero_edges_w_to_be_removed] = False
    top_edges_after_filter_mask[zero_edges_w_to_be_removed] = False
    top_m_edges_w_noisy = top_edges_after_filter[top_edges_after_filter_mask]
    
    utils.log_msg('m = %s/%s, orig. remaining edges = %s/%s (%s) ' % ( len(top_m_edges_w_noisy), len(arr), num_remaining_edges, len(arr), float("{:.2f}".format(num_remaining_edges/len(arr))) ) )

    return top_m_edges_w_noisy, num_remaining_edges

def gradient_descent(init, steps, grad, laplaced_sum, proj=lambda x: x, min_value=1):
    xs = [init]
    for step in steps:
        xs_next = proj(xs[-1] - step * grad(xs[-1]), laplaced_sum, min_value)
        xs = [xs_next]
    return xs

def l2(x, y):
    return np.sum(np.power( ( y - x ),2) )

def l2_gradient(x, y):
    return 2*(y - x)

def proj(x, desired_sum, min_value=1):
    """Projection of x onto the subspace"""
    current_sum = np.sum(x)
    factor = (current_sum - desired_sum)/len(x)
    projection = np.around(np.clip(x - factor, min_value, None))     
    
    return projection

def min_l2_norm(edges_w, desired_sum, num_steps=500, min_value=1):
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

def sample_graph(g, edges_w_noisy_non_zero, edges_w_noisy_zeros_pp, with_ns_property=False):

    edges_w_noisy_zeros = edges_w_noisy_zeros_pp[np.nonzero(edges_w_noisy_zeros_pp)]

    new_g = WGraph()
    new_g.add_vertex(n=g.n())
    optin = new_g.new_vertex_property('bool')
    optin.fa = g.vp.optin.fa
    new_g.vertex_properties['optin'] = optin

    ew = new_g.new_edge_property('int')
    new_g.edge_properties['ew'] = ew
    # new_g.vp.optin = g.vp.optin.copy()
    # new_g.clear_edges()
    
    mask_in_in = g.edges_in_in() 
    edges_in_in = g.get_edges([g.ep.ew])[mask_in_in]
    new_g.add_edge_list(edges_in_in, eprops=[new_g.ep.ew])

    mask_not_in_in = g.edges_without_in_in()
    edges_not_in_in = g.get_edges()[mask_not_in_in]
    non_zero_edges_weighted = np.concatenate((edges_not_in_in, np.array([ edges_w_noisy_non_zero ]).T ), axis=1)
    edges_non_zero_filt =  non_zero_edges_weighted[non_zero_edges_weighted[:,2] != 0 ]  # filter edges_w != 0
    new_g.add_edge_list(edges_non_zero_filt, eprops=[new_g.ep.ew])

    # adjacency_matrix = graph_tool.spectral.adjacency(g, weight=g.ep.we)
    # optins = new_g.vp.optin.fa
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

    if with_ns_property:
        ns = new_g.new_vertex_property('int')

        for v in new_g.vertices():
            ns[v] = new_g.node_strength(v)

        new_g.vertex_properties['ns'] = ns

    return new_g