import os
import numpy as np
import graph_tool.all as gt

from utils import messages as msgs
from numpy import genfromtxt
from graph.wgraph import WGraph

np.random.seed(0)

class statistics():
    
    def __init__(self, datasets_names, optins_method, optins_perc):
        self.datasets_names = datasets_names
        self.optins_method = optins_method
        self.optins_perc = optins_perc

    def show(self):
        for dataset in self.datasets_names:
            for optin_method in self.optins_method: 
                for optin_perc in self.optins_perc:
                    url = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', dataset, '%s_%s_%s.graphml' % (dataset, optin_method, optin_perc )))
                    g = WGraph(url)
                    msgs.log('======= ' + dataset + ' ======= ')
                    msgs.log('vertices:     %s' % g.n())
                    msgs.log('edges:        %s' % g.m())
                    msgs.log('max degree:   %s' % g.max_degree())
                    msgs.log('degrees avg:  %s' % g.avg_degrees())
                    msgs.log('degrees perc: %s' % g.degrees_percentiles())
                    msgs.log('max edge_w:   %s' % g.max_edge_w())
                    msgs.log('edges_w avg:  %s' % g.avg_edges_w())
                    msgs.log('edges_w perc: %s' % g.edges_w_percentiles())

if __name__ == "__main__":
    datasets_names = [
                    'high-school-contacts',
                    'copenhagen-interaction',
                    'reality-call', 
                    'contacts-dublin',
                    'digg-reply',
                    'reality-call',
                    'wiki-talk',
                    'sx-stackoverflow']

    optins_methods = ['random']
    optins_perc = [.2]

    statistics = statistics(datasets_names, optins_methods, optins_perc)
    statistics.show()
    