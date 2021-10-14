from abc import ABC, abstractmethod

import graph_tool as gt
import numpy as np

class WGraph(object):

    PERCENTILES = np.arange(10, 101,10)

    def __init__(self, path=None, G=None, prune=None):
        if path is not None:
            self.G = gt.load_graph(path)
        elif prune is not None:
            self.G = gt.Graph(G, directed=False, prune=prune)
        else:
            self.G = gt.Graph(G, directed=False)

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
        return np.percentile(self.degrees(), self.PERCENTILES).astype(int)   

    def edges_w(self):
        return self.ep.ew.fa.astype(int)
    
    def max_edge_w(self):
        return np.amax(self.edges_w()) 

    def avg_edges_w(self):
        return np.average(self.edges_w())     

    def edges_w_percentiles(self):
        return np.percentile(self.edges_w(), self.PERCENTILES).astype(int)   

    def __getattr__(self, *args, **kwargs):
        return getattr(self.G, *args, **kwargs)