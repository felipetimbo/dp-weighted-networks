import os
import numpy as np
import graph_tool.all as gt

from dpwnets import utils
from numpy import genfromtxt
from graph.wgraph import WGraph

np.random.seed(0)

class Statistics():
    
    def __init__(self, datasets_names, optins_methods, optins_perc):
        self.datasets_names = datasets_names
        self.optins_methods = optins_methods
        self.optins_perc = optins_perc

    def show(self):
        for dataset in self.datasets_names:
            for optin_method in self.optins_methods: 
                for optin_perc in self.optins_perc:
                    url = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', dataset, '%s_%s_%s.graphml' % (dataset, optin_method, optin_perc )))
                    g = WGraph(url)
                    utils.log_msg('======= ' + dataset + ' ======= ')
                    utils.log_msg('vertices:    %s' % g.n())
                    utils.log_msg('edges:       %s' % g.m())
                    utils.log_msg('max degree:  %s' % g.max_degree())
                    utils.log_msg('degrees avg: %s' % g.avg_degrees())
                    utils.log_msg('degrees dec: %s' % g.degrees_percentiles())
                    utils.log_msg('degrees qua: %s' % g.degrees_quartiles())
                    utils.log_msg('max edge_w:  %s' % g.max_edge_w())
                    utils.log_msg('edges_w avg: %s' % g.avg_edges_w())
                    utils.log_msg('edges_w dec: %s' % g.edges_w_percentiles())    
                    utils.log_msg('edges_w qua: %s' % g.edges_w_quartiles())    