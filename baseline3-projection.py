
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
                    thetas,
                    es, 
                    runs
                ):
        self.datasets_names = datasets_names  
        self.optins_methods = optins_methods
        self.optins_perc = optins_perc
        self.thetas = thetas
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

                    # mask_out_out = g.edges_out_out()
                    # g_out_out = WGraph(G=gt.GraphView(g, efilt=mask_out_out), prune=True)

                    len_all_edges_without_in_in = int( ( g.n() * (g.n()-1))/2 - (len(optins) * (len(optins)-1))/2 )

                    for e in self.es:
                        utils.log_msg('******* eps = ' + str(e) + ' *******')

                        for theta in self.thetas:
                            utils.log_msg('***** theta = ' + str(theta) + ' *****')

                            # projection
                            edges = g.get_edges([g.ep.ew]) 
                            projected_edges_w = np.clip(edges[:,2], 1, theta) 
                            projected_edges = np.concatenate((edges[:,[0,1]], np.array([projected_edges_w]).T ), axis=1)

                            g_projected = tools.build_g_from_edges(g, projected_edges)
                            edges = g_projected.get_edges([g_projected.ep.ew]) 

                            for r in range(self.runs):
                                utils.log_msg('... RUN ' + str(r) + ' ...')   
                                
                                edges_w_noisy = dp_mechanisms.geometric_mechanism(edges[:,2], e, theta) 
                                edges_w_noisy_clipped = np.clip(edges_w_noisy, 1, None) 
                                new_edges = np.concatenate((edges[:,[0,1]], np.array([edges_w_noisy_clipped]).T ), axis=1)

                                new_g = tools.build_g_from_edges(g_projected, new_edges)

                                utils.log_msg('saving graph...')
                                path_graph = "./data/%s/exp/%s_ins%s_e%s_t%s_r%s_baseline3.graphml" % ( dataset , optin_method, optin_perc, e, theta, r)     
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

    thetas = [ 13 ]

    runs = 10

    exp = DPWeightedNets(datasets_names, optins_methods, optins_perc, thetas, es, runs)
    exp.run()