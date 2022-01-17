from abc import ABC, abstractmethod
from graph_tool.topology import shortest_distance

import graph_tool as gt
import numpy as np

class WGraph(object):

    def __init__(self, path=None, G=None, prune=None, compute_distances=False):
        if path is not None:
            self.G = gt.load_graph(path)
        elif prune is not None:
            self.G = gt.Graph(G, directed=False, prune=prune)
        else:
            self.G = gt.Graph(G, directed=False)
        if compute_distances:
            distances = np.array([], dtype='int')
            for v in self.G.vertices():
                s_paths = shortest_distance(self.G, v).fa.astype('int')
                s_paths = s_paths[s_paths != 2147483647]
                s_paths = s_paths[s_paths != 0]
                distances = np.append(distances, s_paths, axis=0)  
            self.distances = distances

    def n(self):
        return self.G.num_vertices()

    def m(self):
        return self.G.num_edges()

    def degrees(self):
        return self.G.get_out_degrees(self.G.get_vertices())

    def max_degree(self, cached=False):
        return np.amax(self.degrees()) 

    def avg_degrees(self):
        return np.average(self.degrees()) 

    def degrees_percentiles(self):
        deciles = np.arange(10, 101,10)
        return np.percentile(self.degrees(), deciles).astype(int)   

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

    def edges_w_percentiles(self):
        deciles = np.arange(10, 101,10)
        return np.percentile(self.edges_w(), deciles).astype(int)   

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

    def __getattr__(self, *args, **kwargs):
        return getattr(self.G, *args, **kwargs)