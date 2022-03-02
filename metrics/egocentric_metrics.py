import numpy as np
import pandas as pd
import graph_tool
import multiprocessing

from graph_tool.centrality import betweenness, pagerank, eigenvector
from graph_tool.topology import pseudo_diameter, similarity, shortest_distance, max_cliques, extract_largest_component, shortest_path
from dpwnets import tools

import graph_tool.all as gt

def calculate(G, metric):
    if metric == 'degree':
        optins_arr = degree(G)
    elif metric == 'degree_all':
        optins_arr = degree(G, only_optins=False)
    elif metric == 'att_influence':
        optins_arr = att_influence(G)   
    elif metric == 'node_strength':
        optins_arr = node_strength(G)   
    elif metric == 'node_strength_all':
        optins_arr = node_strength(G, only_optins=False)
    elif metric == 'node_edges_weight_avg':
        optins_arr = node_edges_weight_avg(G)   
    elif metric == 'node_edges_weight_avg_all':
        optins_arr = node_edges_weight_avg(G, only_optins=False)
    elif metric == 'sum_of_2_hop_edges':
        optins_arr = sum_of_2_hop_edges(G)
    elif metric == 'sum_of_2_hop_edges_all':
        optins_arr = sum_of_2_hop_edges(G, only_optins=False)
    elif metric == 'num_edges_in_alters':
        optins_arr = num_edges_in_alters(G)
    elif metric == 'num_edges_in_alters_all':
        optins_arr = num_edges_in_alters(G, only_optins=False)
    elif metric == 'ego_betweenness':
        optins_arr = ego_betweenness(G)
    elif metric == 'ego_betweenness_all':
        optins_arr = ego_betweenness(G, only_optins=False)
    elif metric == 'ego_betweenness_w':
        optins_arr = ego_betweenness_w(G)
    elif metric == 'ego_betweenness_w_all':
        optins_arr = ego_betweenness_w(G, only_optins=False)
    elif metric == 'density': 
        optins_arr = density(G)
    elif metric == 'density_all': 
        optins_arr = density(G, only_optins=False)
    elif metric == 'density_w': 
        optins_arr = density_w(G)
    elif metric == 'density_w_all': 
        optins_arr = density_w(G, only_optins=False)
    elif metric == 'pagerank': 
        optins_arr = page_rank(G)
    elif metric == 'pagerank_all': 
        optins_arr = page_rank(G, only_optins=False)
    elif metric == 'pagerank_w': 
        optins_arr = page_rank_w(G)
    elif metric == 'pagerank_w_all': 
        optins_arr = page_rank_w(G, only_optins=False)
    elif metric == 'betweenness_w': 
        optins_arr = betweenness_w(G)
    elif metric == 'betweenness_w_all': 
        optins_arr = betweenness_w(G, only_optins=False)
    elif metric == 'eigenvector_w': 
        optins_arr = eigenvector_w(G)
    elif metric == 'eigenvector_w_all': 
        optins_arr = eigenvector_w(G, only_optins=False)
    elif metric == 'local_clustering_w': 
        optins_arr = local_clustering_w(G)
    elif metric == 'local_clustering_w_all': 
        optins_arr = local_clustering_w(G, only_optins=False)
    elif metric == 'global_clustering_w': 
        optins_arr = global_clustering_w(G)
    elif metric == 'global_clustering_w_all': 
        optins_arr = global_clustering_w(G, only_optins=False)
    elif metric == 'm':
        optins_arr = m(G)
    elif metric == 'total_w':
        optins_arr = total_w(G)
    elif metric == 'edges_w':
        optins_arr = edges_w(G)
    elif metric == 'diameter':
        optins_arr = diameter(G)
    elif metric == 'diam':
        optins_arr = diam(G)
    elif metric == 'avg_shortest_path':
        optins_arr = avg_shortest_path(G)
    elif metric == 'avg_shortest_path_w':
        optins_arr = avg_shortest_path_w(G)
    elif metric == 'density_G':
        optins_arr = density_G(G) 
    elif metric == 'avg_degree':
        optins_arr = avg_degree(G)
    elif metric == 'avg_edges_w':
        optins_arr = avg_edges_w(G)
    elif metric == 'strong_triangles':
        optins_arr = strong_triangles(G) 
    elif metric == 'strong_triangles2':
        optins_arr = strong_triangles2(G)  
    elif metric == 'strong_shortest_paths':
        optins_arr = strong_shortest_paths(G)  
    else:
        optins_arr = None

    return optins_arr       

def diameter(G):
    return G.diameter()

def diam(G):
    return pseudo_diameter(G)[0]

def avg_shortest_path(G):
    return G.avg_shortest_path()

def avg_shortest_path_w(G):
    return G.avg_shortest_path_w()

def density_G(G):
    return G.density()

def avg_degree(G):
    return G.avg_degrees()

def avg_edges_w(G):
    return G.avg_edges_w()

def similar(G1, G2):
    return similarity(G1, G2, eweight1=G1.ep.ew, eweight2=G2.ep.ew)

def m(G):
    return [G.m()]

def total_w(G):
    return [np.sum(G.ep.ew.fa)]

def edges_w(G):
    return G.get_edges([G.ep.ew])

def ego_edges_w(G, only_optins=True):
    E = []

    if only_optins:
        vertices = G.optins() 
    else:
        vertices = G.vertices()

    for v_id in vertices:
        edges_v = G.get_out_edges(v_id, [G.ep.ew] )
        E.append(edges_v)
        # egonet = get_ego_network(G, v_id)
        # g = EgoGraph(G=egonet, prune=True)
        # E.append(g.get_edges([g.ep.ew]))

        # return np.array(G.ep.ew.fa)
    return E

def degree(G, only_optins=True):
    D = [] 
    if only_optins:
        vertices = G.optins() 
    else:
        vertices = G.vertices()

    for v_id in vertices:
        D.append(G.vertex(v_id).out_degree())

    return np.array(D) 

def node_strength(G, only_optins=True):
    S = []

    if only_optins:
        vertices = G.optins() 
    else:
        vertices = G.vertices()

    for v_id in vertices:
        edges_v = G.get_out_edges(v_id, [G.ep.ew] )
        strength = np.sum(edges_v[:,2]) 
        S.append(strength)

    return np.array(S) 
        
def node_edges_weight_avg(G, only_optins=True):
    A = []

    if only_optins:
        vertices = G.optins() 
    else:
        vertices = G.vertices()

    for v_id in vertices:
        edges_v = G.get_out_edges(v_id, [G.ep.ew] )
        if len(edges_v) > 0:
            strength = np.average(edges_v[:,2]) 
        else:
            strength = 0
        A.append(strength)

    return np.array(A) 

# def degree_all(G):
#     D = [] #np.zeros((G.n()))
#     for v_id in G.vertices():
#         # D[v_id] = v_id.out_degree() new_graph.edge
#         D.append(G.vertex(v_id).out_degree())

#     return np.array(D) 

def att_influence(G):
    I = [] # np.zeros((G.n()))
    for v_id in G.optins():
        neighbors_id = G.get_out_neighbors(v_id)
        neighbors_deg = G.get_out_degrees(neighbors_id)
        # I[v_id] = sum(neighbors_deg)
        I.append(sum(neighbors_deg))

    return np.array(I)

def sum_of_2_hop_edges_bkp(G):
    return att_influence_hops(G, 2, 2, 1)

def sum_of_2_hop_edges(G, only_optins=True):
    I = []

    ns = G.new_vertex_property('int')
    G.vertex_properties['ns'] = ns
    gt.incident_edges_op(G, "out", "sum", G.ep.ew, G.vp.ns)

        # for v_id in G.optins():
        #     ego_net_2_hop = get_ego_network_2_hop(G, v_id)       
        #     edges_w = np.array(ego_net_2_hop.ep.ew.fa)
        #     infl_v = np.sum(edges_w)
        #     I.append(infl_v)

    if only_optins:
        for v_id in G.optins():
            infl_v = 1
            for _, i in G.iter_all_neighbors(v_id, [G.vp.ns]):
                infl_v += i
            # ego_net = get_ego_network(G, v_id)
            # ns = np.array(ego_net.vp.ns.fa) 
            # infl_v = np.sum(ns) 
            I.append(infl_v)
    else:
        for v_id in G.vertices():
            ego_net = get_ego_network(G, v_id)
            ns = np.array(ego_net.vp.ns.fa) 
            infl_v = np.sum(ns) # - G.node_strength(v_id)
            I.append(infl_v)

    return np.array(I)

def att_influence_hops(G, w1, w2, w3):
    I = []

    for v_id in G.optins():
        # print('opt-in: ' + str(v_id))

        # edges 1-hop
        edges_1_hop = []
        hop_1_w = 0
        for e in G.vertex(v_id).out_edges():
            edges_1_hop.append(e)
            hop_1_w += G.ep.ew[e]
        # print('weight_hop_1: ' + str(hop_1_w))

        # edges 1.5-hop
        ego_net = get_ego_network(G, v_id)
        edges_1_5_hop = []
        hop_1_5_w = 0
        for e in ego_net.edges():
            if e not in (edges_1_hop):
                edges_1_5_hop.append(e)
                hop_1_5_w += G.ep.ew[e]
        # print('weight_hop_1.5: ' + str(hop_1_5_w))

        # edges 2-hop
        neighbors_id = G.get_out_neighbors(v_id)
        edges_2_hop = []
        hop_2_w = 0
        for neighbor_id in neighbors_id:
            for e in G.vertex(neighbor_id).out_edges():
                if(e not in edges_1_hop):
                    if(e not in edges_1_5_hop):
                        edges_2_hop.append(e)
                        hop_2_w += G.ep.ew[e]
        # print('weight_hop_2: ' + str(hop_2_w))

        infl_v = hop_1_w*w1 + hop_1_5_w*w2 + hop_2_w*w3
        # print('infl_v' + str(v_id) + ': ' + str(infl_v))

        # I[v_id] = infl_v
        I.append(infl_v)

    return np.array(I)

def num_edges_in_alters(G, only_optins=True):
    E = [] 

    if only_optins:
        vertices = G.optins() 
    else:
        vertices = G.vertices()
        
    for v_id in vertices:
        egonet = get_ego_network(G, v_id)
        num_alter_edges = alter_edges(egonet, v_id)
        E.append(num_alter_edges)

    return np.array(E)

def ego_betweenness(G, only_optins=True):
    B = [] # np.zeros((G.n()))

    if only_optins:
        vertices = G.optins() 
    else:
        vertices = G.vertices()

    for v_id in vertices:
        egonet = get_ego_network(G, v_id)
        bt = ego_bt(egonet, v_id)
        B.append(bt)

    return np.array(B)

def ego_betweenness(G, only_optins=True):
    B = [] # np.zeros((G.n()))

    if only_optins:
        vertices = G.optins() 
    else:
        vertices = G.vertices()

    for v_id in vertices:
        egonet = get_ego_network(G, v_id)
        bt = ego_bt(egonet, v_id)
        B.append(bt)

    return np.array(B)

def ego_betweenness_w(G):
    B = [] # np.zeros((G.n()))
    for v_id in G.optins():
        egonet = get_ego_network(G, v_id)
        b, _ = betweenness(egonet, weight=egonet.ep.ew, norm=False)
        bt = b[v_id]
        B.append(bt)

    return np.array(B)

def ego_page_rank(G, only_optins=True):
    PR = [] 

    if only_optins:
        vertices = G.optins() 
    else:
        vertices = G.vertices()

    for v_id in vertices:
        egonet = get_ego_network(G, v_id)
        rank = pagerank(egonet)
        PR.append(rank[v_id])

    return np.array(PR)

# def random_walk(G, only_optins=True):
#     PR = [] 

#     if only_optins:
#         vertices = G.optins() 
#     else:
#         vertices = G.vertices()

#     for v_id in vertices:
#         egonet = get_ego_network(G, v_id)
#         g = EgoGraph(G=egonet)
#         rank = eigenvector(g, weight=g.ep.ew)
#         PR.append(rank[1][v_id])

#     return np.array(PR)

def page_rank_w(G, only_optins=True):
    if only_optins:
        return np.array(pagerank(G).fa.astype('float'))[G.optins()]
    else:
        return np.array(pagerank(G).fa.astype('float'))

def betweenness_w(G, only_optins=True):
    if only_optins:
        return np.array(betweenness(G)[0].fa.astype('float'))[G.optins()]
    else:
        return np.array(betweenness(G)[0].fa.astype('float'))

def eigenvector_w(G, only_optins=True):
    if only_optins:
        return np.array(eigenvector(G)[1].fa.astype('float'))[G.optins()]
    else:
        return np.array(eigenvector(G)[1].fa.astype('float'))


def density(G, only_optins=True):
    D = [] # np.zeros((G.n()))

    if only_optins:
        vertices = G.optins() 
    else:
        vertices = G.vertices()

    # gt.clustering.local_clustering(G)

    for v_id in vertices:
        egonet = get_ego_network(G, v_id)
        density = clustering_coefficient(egonet, v_id)
        D.append(density)

    return np.array(D)


def local_clustering_w(G, only_optins=True):
    if only_optins:
        return np.array(graph_tool.clustering.local_clustering(G,weight=G.ep.ew).fa.astype('int'))[G.optins()]
    else:
        return np.array(graph_tool.clustering.local_clustering(G,weight=G.ep.ew).fa.astype('int'))

def global_clustering_w(G, only_optins=True):
    if only_optins:
        return np.array(graph_tool.clustering.global_clustering(G,weight=G.ep.ew))[0]
    else:
        return np.array(graph_tool.clustering.global_clustering(G,weight=G.ep.ew))[0]

def get_ego_network(G, v, prune=False):
    edges_v = G.get_out_edges(v)
    neighbors_v = edges_v[:,1]
    mask = np.full((G.num_vertices(),), False)
    mask[neighbors_v] = True
    mask_p = G.new_vertex_property("bool")
    mask_p.fa = mask
    mask_p[v] = True
    egonet = gt.GraphView(G, vfilt=mask_p)
    # pos = gt.arf_layout(egonet, max_iter=0)
    # gt.graph_draw(egonet, pos=pos, output="lattice-planar.pdf")
    # vp, _ = betweenness(Gv, norm=False)
    # G.vp.I[v] = vp[v]
    
    # return EgoGraph(G=egonet, prune=True) #
    return egonet

def get_ego_network_2_hop(G, v):
    edges_v = G.get_out_edges(v)
    neighbors_v = edges_v[:,1]
    mask = np.full((G.num_vertices(),), False)
    mask[neighbors_v] = True

    all_neighbors_d_2 = set()
    for n_id in neighbors_v:
        neighbors_n = G.get_out_edges(n_id)[:,1]
        mask[neighbors_n] = True
        neighbors_d_2 = set(neighbors_n).difference( {v}.union(neighbors_v) )
        all_neighbors_d_2 = all_neighbors_d_2.union(neighbors_d_2)

    mask_p = G.new_vertex_property("bool")
    mask_p.fa = mask   
    mask_p[v] = True

    all_neighbors_d_2_list = list(all_neighbors_d_2)
    mask_edges = np.full((G.num_edges(),), True) 
    edges = G.get_edges() 
    sources = edges[:,0]
    sources_mask = np.in1d(sources, all_neighbors_d_2_list)
    targets = edges[:,1] 
    targets_mask = np.in1d(targets, all_neighbors_d_2_list)
    
    edges_mask_ = sources_mask & targets_mask
    edges_mask = ~edges_mask_

    egonet = gt.GraphView(G, vfilt=mask_p, efilt=edges_mask)
    return egonet

def alter_edges(egonet, v):
    return egonet.num_edges() - egonet.vertex(v).out_degree()

def clustering_coefficient(egonet, v):
    d = egonet.vertex(v).out_degree()
    if d == 0 or d == 1:
        return 0
    else:
        # return (alter_edges(egonet, v))/(d*(d-1))
        return (egonet.num_edges() - d)/((d*(d-1)/2))

def ego_bt(egonet, v):
    b, _ = betweenness(egonet, norm=False)
    return b[v]

def ego_bt_w(egonet, v):
    b, _ = betweenness(egonet, weight=egonet.ep.ew, norm=False)
    return b[v]

def strong_triangles2(G):
    triangles = []
    for c in gt.max_cliques(G):
        if len(c) == 3:
            e1 = G.edge(c[0],c[1])
            e2 = G.edge(c[0],c[2])
            e3 = G.edge(c[1],c[2])
            w1 = G.ep.ew[e1]
            w2 = G.ep.ew[e2]
            w3 = G.ep.ew[e3]
            triangle_strength = w1+w2+w3
            c_sorted = sorted(c)
            triangle = np.append(c_sorted, triangle_strength)
            triangles.append(triangle)

    df_triangles = pd.DataFrame(data=np.array(triangles), columns=["n1", "n2", "n3", "w"], dtype="int")
    df_triangles = df_triangles.sort_values(by=['w'], ascending=False)
    tringles_arr = df_triangles.to_numpy()

    return tringles_arr

def strong_triangles(G):
    triangles = []
    edges_w = G.ep.ew.fa.astype('int')
    threshold = np.percentile(edges_w, 15) 
    for c in gt.max_cliques(G):
        if len(c) == 3: 
            e1 = G.edge(c[0],c[1])
            e2 = G.edge(c[0],c[2])
            e3 = G.edge(c[1],c[2])
            w1 = G.ep.ew[e1]
            w2 = G.ep.ew[e2]
            w3 = G.ep.ew[e3]
            if w1 > threshold and w2 > threshold and w3 > threshold:
                triangle_strength = w1+w2+w3
                c_sorted = sorted(c)
                triangle = np.append(c_sorted, triangle_strength)
                triangles.append(triangle)

    if len(triangles) > 0:
        df_triangles = pd.DataFrame(data=np.array(triangles), columns=["n1", "n2", "n3", "w"], dtype="int")
        df_triangles = df_triangles.sort_values(by=['w'], ascending=False)
        tringles_arr = df_triangles.to_numpy()
    else:
        tringles_arr = []

    return tringles_arr

def strong_shortest_paths(G):
    edges_w = G.ep.ew.fa.astype('int')
    threshold = np.percentile(edges_w, 85) 
    new_edges = tools.filter_edges_higher_than_threshold(G, threshold)
    new_g = tools.build_g_from_edges(G, new_edges)
    largest_component_extracted = extract_largest_component(new_g)
    s_paths = compute_distances(largest_component_extracted)
    return s_paths

def compute_distances(g):

    num_threads = 15
    vertices = list(g.get_vertices())
    range_v = np.array_split(vertices, num_threads) # np.array_split(list(range(g.num_vertices())), num_threads)
    distances = np.array([], dtype='int')
    manager = multiprocessing.Manager() 
    distances_multiprocessing = manager.list() 
    # all_distances_multiprocessing = manager.list() 
    threads = []
    for i in range(num_threads):
        t = multiprocessing.Process( target = compute_distances_parallel, args =(g, range_v[i], distances_multiprocessing, vertices ))
        t.start()
        threads.append(t)
    for t in threads:    
        t.join() 
    distances = list(distances_multiprocessing)
    distances_df = pd.DataFrame(distances)
    distances_sorted_df = distances_df.sort_values(by=[0], ascending=False).fillna(np.inf)
    distances_arr = distances_sorted_df.to_numpy()

    return distances_arr
    # print(new_g)

def compute_distances_parallel(g, range_v, distances,  all_vertices):
    for v in range_v:
        for u in all_vertices:
            if v < u:
                s_dist = shortest_distance(g, v, u, weights=g.ep.ew) 
                if s_dist != 2147483647:
                    # all_distances.append(s_dist)
                    vlist, elist = shortest_path(g, v, u, weights=g.ep.ew) 
                    s_path = [int(v) for v in vlist]
                    weight_path = np.append(s_dist, s_path).astype('int').tolist()
                    distances.append(weight_path)  

# def effective_size(egonet, v):
#     return degree(v) - (2*alter_edges(egonet, v))/degree(v)

# def efficiency(egonet, v):
#     return effective_size(egonet, v)/degree(v)

