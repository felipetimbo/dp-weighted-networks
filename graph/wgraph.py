from abc import ABC, abstractmethod
from graph_tool.topology import (shortest_distance, shortest_path)

import graph_tool as gt 
import graph_tool.all as gt_
import pandas as pd
import numpy as np
import multiprocessing

from dpwnets import utils

class WGraph(object):

    def __init__(self, path=None, G=None, prune=None, compute_distances=False):
        if path is not None:
            self.G = gt.load_graph(path)
        elif prune is not None:
            self.G = gt.Graph(G, directed=False, prune=prune)
        else:
            self.G = gt.Graph(G, directed=False)

        # if compute_distances:
        #     num_threads = 15
        #     thresold = 100
        #     range_v = np.array_split(list(range(self.G.num_vertices())), num_threads)
        #     distances = np.array([], dtype='int')
            
        #     manager = multiprocessing.Manager() 
        #     distances_multiprocessing = manager.list() 
        #     # all_distances_multiprocessing = manager.list() 

        #     threads = []
        #     for i in range(num_threads):
        #         t = multiprocessing.Process( target = self.compute_distances_parallel, args =(self.G, range_v[i], distances_multiprocessing, list(range(self.G.num_vertices())), thresold ))
        #         t.start()
        #         threads.append(t)
            
        #     for t in threads:    
        #         t.join() 
                   
        #     distances = list(distances_multiprocessing)
        #     distances_df = pd.DataFrame(distances)
        #     distances_sorted_df = distances_df.sort_values(by=[0], ascending=False)
        #     distances_arr = distances_sorted_df.to_numpy()
        #     self.shortests_paths = distances_arr

            # all_distances = np.array(list(all_distances_multiprocessing))
            # self.distances = all_distances
            # self.distances = distances_arr[:,0]

    # def compute_distances_parallel(self, g, range_v, distances,  all_vertices, thresold):
    #     for v in range_v:
    #         for u in all_vertices:
    #             if v != u:
    #                 s_dist = shortest_distance(g, v, u, weights=g.ep.ew) 
    #                 if s_dist != 2147483647:
    #                     # all_distances.append(s_dist)
    #                     if s_dist > thresold:
    #                         vlist, elist = shortest_path(g, v, u, weights=g.ep.ew) 
    #                         s_path = [int(v) for v in vlist]
    #                         weight_path = np.append(s_dist, s_path).astype('int').tolist()
    #                         distances.append(weight_path)  

    def n(self):
        return self.G.num_vertices()

    def m(self):
        return self.G.num_edges()

    def degrees(self):
        return self.G.get_out_degrees(self.G.get_vertices()).astype('int')

    def max_degree(self, cached=False):
        return np.amax(self.degrees()) 

    def avg_degrees(self):
        return np.average(self.degrees()) 

    def degrees_percentiles(self):
        deciles = np.arange(10, 101,10)
        return np.percentile(self.degrees(), deciles).astype(int)   

    def degrees_quartiles(self):
        quartiles = np.array([0, 25, 50, 75, 100])
        return np.percentile(self.degrees(), quartiles).astype(int)   

    def edges_w(self):
        return self.ep.ew.fa.astype(int)
    
    def max_edge_w(self):
        return np.amax(self.edges_w()) 

    def avg_edges_w(self):
        return np.average(self.edges_w())     

    def node_strength(self, v_id):
        edges_v = self.G.get_out_edges(v_id, [self.G.ep.ew] )
        return np.sum(edges_v[:,2]) 

    def node_strengths(self):
        return gt.incident_edges_op(self.G, "out", "sum", self.G.ep.ew).fa.astype(int)

    def sum_2_hop_edges(self):
        sum_2_hop = []
        nss = self.node_strengths()
        for node in range(self.G.num_vertices()):
            neighbors_node = self.G.get_out_neighbors(node)
            all_nodes = np.append(neighbors_node, node)
            sum_2_hop.append( np.sum( nss[all_nodes] ) )

        return np.array(sum_2_hop)

        # ns = G.new_vertex_property('int')
        # self.G.vertex_properties['ns'] = ns
        # gt_.incident_edges_op(G, "out", "sum", G.ep.ew, G.vp.ns)

        # sum_2_hop = []
        # for v_id in range(self.G.vertices()):
        #     ego_net = get_ego_network(elf.G, v_id)
        #     ns = np.array(ego_net.vp.ns.fa) 
        #     infl_v = np.sum(ns) # - G.node_strength(v_id)
        #     sum_2_hop.append(infl_v)

    def edges_w_percentiles(self):
        deciles = np.arange(10, 101,10)
        return np.percentile(self.edges_w(), deciles).astype(int)   

    def edges_w_quartiles(self):
        quartiles = np.array([0, 25, 50, 75, 100])
        return np.percentile(self.edges_w(), quartiles).astype(int)   

    def optins(self):
        optins_mask = self.vp.optin.fa.astype(bool)
        return self.G.get_vertices()[optins_mask]  

    def optouts(self):
        optouts_mask = ~self.vp.optin.fa.astype(bool)
        return self.G.get_vertices()[optouts_mask]

    def edges_without_in_in(self):
        edges = self.G.get_edges()
        sources = edges[:,0]
        sources_optin = self.G.vp.optin.a[sources].astype(bool)
        targets = edges[:,1]
        targets_optin = self.G.vp.optin.a[targets].astype(bool)
        not_optin_edges_mask = np.logical_or(np.invert(sources_optin), np.invert(targets_optin))
        return not_optin_edges_mask

    def edges_in_out(self):
        edges = self.G.get_edges()
        sources = edges[:,0]
        sources_optin = self.G.vp.optin.a[sources].astype(bool)
        targets = edges[:,1]
        targets_optin = self.G.vp.optin.a[targets].astype(bool) 
        optinout_edges_mask = np.logical_and(np.invert(sources_optin), targets_optin) | np.logical_and(sources_optin, np.invert(targets_optin))
        return optinout_edges_mask

    def edges_out_out(self):
        edges = self.G.get_edges()
        sources = edges[:,0]
        sources_optin = self.G.vp.optin.a[sources].astype(bool)
        targets = edges[:,1]
        targets_optin = self.G.vp.optin.a[targets].astype(bool)
        optout_edges_mask = np.invert(sources_optin) & np.invert(targets_optin)
        return optout_edges_mask

    def edges_in_in(self):
        edges = self.G.get_edges()
        sources = edges[:,0]
        sources_optin = self.G.vp.optin.a[sources].astype(bool)
        targets = edges[:,1]
        targets_optin = self.G.vp.optin.a[targets].astype(bool)
        optin_edges_mask = sources_optin & targets_optin
        return optin_edges_mask

    def diameter(self):
        return max(self.distances)

    def avg_shortest_path(self):
        return np.mean(self.distances)

    def avg_shortest_path_w(self):
        d = np.array([], dtype='int')
        for v in self.G.vertices():
            s_paths = shortest_distance(self.G, v, weights=self.G.ep.ew).fa.astype('int')
            s_paths = s_paths[s_paths != 2147483647]
            s_paths = s_paths[s_paths != 0]
            d = np.append(d, s_paths, axis=0)  
        return np.mean(d)

    def density(self):
        return 2*self.m()/self.n()*(self.n() - 1)

    def get_ego_network(self, v):
        edges_v = self.G.get_out_edges(v)
        neighbors_v = edges_v[:,1]
        mask = np.full((G.num_vertices(),), False)
        mask[neighbors_v] = True
        mask_p = G.new_vertex_property("bool")
        mask_p.fa = mask
        mask_p[v] = True
        egonet = gt_.GraphView(G, vfilt=mask_p)
        return egonet

    def print_statistics(self):
        utils.log_msg('vertices:    %s' % self.n())
        utils.log_msg('edges:       %s' % self.m())
        utils.log_msg('max degree:  %s' % self.max_degree())
        utils.log_msg('degrees avg: %s' % self.avg_degrees())
        utils.log_msg('degrees dec: %s' % self.degrees_percentiles())
        utils.log_msg('degrees qua: %s' % self.degrees_quartiles())
        utils.log_msg('max edge_w:  %s' % self.max_edge_w())
        utils.log_msg('edges_w avg: %s' % self.avg_edges_w())
        utils.log_msg('edges_w dec: %s' % self.edges_w_percentiles())    
        utils.log_msg('edges_w qua: %s' % self.edges_w_quartiles()) 

    def __getattr__(self, *args, **kwargs):
        return getattr(self.G, *args, **kwargs)