
import os
import math
import numpy as np
import graph_tool as gt

from dpwnets import utils
from dpwnets import dp_mechanisms
from dpwnets import tools

from metrics import (error_metrics, egocentric_metrics)

from graph.wgraph import WGraph

np.random.seed(1) 

class DPWeightedNets():

    def __init__(self, 
                    datasets_names, 
                    optins_methods, 
                    optins_perc, 
                    es, 
                    runs
                ):
        self.datasets_names = datasets_names  
        self.optins_methods = optins_methods
        self.optins_perc = optins_perc
        self.es = es
        self.runs = runs 

    def run(self):
        for dataset in self.datasets_names:
            utils.log_msg('*************** DATASET = ' + dataset + ' ***************')

            for optin_method in self.optins_methods: 
                for optin_perc in self.optins_perc:
                    url = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, '%s_%s_%s.graphml' % (dataset, optin_method, optin_perc )))
                    g = WGraph(url)

                    n = g.n()
                    optins = g.optins()
                    optouts = g.optouts()
                    non_optins_pos = g.vp.optin.fa == 0

                    mask_without_in_in = g.edges_without_in_in()
                    g_without_in_in = WGraph(G=gt.GraphView(g, efilt=mask_without_in_in), prune=True)

                    edges = g.get_edges([g.ep.ew]) 

                    # mask_out_out = g.edges_out_out()
                    # g_out_out = WGraph(G=gt.GraphView(g, efilt=mask_out_out), prune=True)

                    len_all_edges_without_in_in = int( ( g.n() * (g.n()-1))/2 - (len(optins) * (len(optins)-1))/2 )

                    sensitivity = max(edges[:,2])

                    for e in self.es:
                        utils.log_msg('******* eps = ' + str(e) + ' *******')

                        for r in range(self.runs):
                            utils.log_msg('....... RUN ' + str(r) + ' .......')   
                            
                            edges_w_noisy = dp_mechanisms.log_laplace_mechanism(edges[:,2], e) 
                            edges_w_noisy_clipped = np.clip(edges_w_noisy, 1, None) 
                            new_edges = np.concatenate((edges[:,[0,1]], np.array([edges_w_noisy_clipped]).T ), axis=1)

                            new_g = tools.build_g_from_edges(g, new_edges)

                            utils.log_msg('saving graph...')
                            path_graph = "./data/%s/exp/%s_ins%s_e%s_r%s_baseline2.graphml" % ( dataset , optin_method, optin_perc, e, r)     
                            new_g.save(path_graph)                            

if __name__ == "__main__":
    datasets_names = [
                          'high-school-contacts',
                          'reality-call2',
                          'enron',
                          'dblp'
                    ]

    optins_methods = ['affinity']
    optins_perc = [.0]

    es = [ .1, .5, 1 ]

    runs = 10

    exp = DPWeightedNets(datasets_names, optins_methods, optins_perc, es, runs)
    exp.run()