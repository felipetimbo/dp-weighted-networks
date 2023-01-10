
import os
import math
import mpmath
import numpy as np
import graph_tool as gt
import pickle as pkl

from dpwnets import utils
from dpwnets import dp_mechanisms
from dpwnets import tools
from dpwnets import graphics

from metrics import (error_metrics, egocentric_metrics)

from graph.wgraph import WGraph

np.random.seed(1) 

class DPWeightedNets():

    def __init__(self, 
                    datasets_names, 
                    optins_methods, 
                    optins_perc, 
                    levels_length,
                    num_threads,
                    es, 
                    runs
                ):
        self.datasets_names = datasets_names  
        self.optins_methods = optins_methods
        self.optins_perc = optins_perc
        self.levels_length = levels_length
        self.num_threads = num_threads
        self.es = es
        self.runs = runs 

    def find_point_of_decreasing(self):
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

                    len_all_edges_without_in_in = int( ( g.n() * (g.n()-1))/2 - (len(optins) * (len(optins)-1))/2 )

                    utils.log_msg('find_point_of_decreasing...')

                    for e in self.es:
                        utils.log_msg('******* eps = ' + str(e) + ' *******')
                        level = dp_mechanisms.find_point_of_decreasing(len_all_edges_without_in_in, e)
                        utils.log_msg('level with max prob %s...' % str(level) )
                        with open("./data/%s/levels/level_max_prob_e%s.pkl" % ( dataset, e ), 'wb') as f:
	                        pkl.dump(level, f) 


    def pre_process(self):
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

                    # edges_w = g_without_in_in.edges_w()
                    # num_total_edges = int(g_without_in_in.n()*(g_without_in_in.n()-1)/2)
                    # all_edges_w = np.append(edges_w, [0] * (num_total_edges - len(edges_w)))

                    # all_edges_w = np.sort( np.random.choice(all_edges_w, 100, replace=False) )[::-1]

                    # utils.log_msg('computing levels...')
                    # levels_size = dp_mechanisms.get_levels_size(all_edges_w, self.levels_range, self.num_threads)

                    # picked_levels = []
                    # prob_masses = []
                    # graphic_pos = np.linspace(1, self.levels_range[1]-2 - self.levels_range[0], num=1000, dtype=int)
                    # graphic_pos = np.linspace(0, len(levels_size)-2, num=1000, dtype=int)

                    for e in self.es:
                        utils.log_msg('******* eps = ' + str(e) + ' *******')

                        with open("./data/%s/levels/level_max_prob_e%s.pkl" % ( dataset, e ), 'rb') as f:
	                        level_with_max_prob = pkl.load(f)

                        levels_range = [ level_with_max_prob - self.levels_length, level_with_max_prob + self.levels_length ]
                        levels_size = dp_mechanisms.get_levels_size(len_all_edges_without_in_in, levels_range, self.num_threads)

                        prob_mass = dp_mechanisms.compute_probability_mass(levels_range, levels_size, e)
                        with open("./data/%s/levels/prob_mass_e%s.pkl" % ( dataset, e ), 'wb') as f:
	                        pkl.dump(prob_mass, f) 

                        # prob_masses.append(np.array(prob_mass)[graphic_pos])

                        for r in range(self.runs):
                            # utils.log_msg('....... RUN ' + str(r) + ' .......')   
                            
                            # utils.log_msg('running exponential...')
                            picked_level = dp_mechanisms.exponential_mechanism(levels_range, prob_mass)

                            utils.log_msg('picked level: %s' % picked_level)

                    # path_graphic = "./data/%s/levels/prob_mass.png" % ( dataset )

                    # legends = [
                    #             '$\epsilon$=0.1',
                    #             '$\epsilon$=0.5',
                    #             '$\epsilon$=1.0'
                    #         ]
                    # graphics.line_plot2( np.array(graphic_pos + self.levels_range[0]), prob_masses,
                    #                         xlabel='level', ylabel= 'prob. mass', ylog=True, #xlim=(0, 25000),
                    #                         path=path_graphic, line_legends=legends, figsize=(10, 5), colors = ['#000000', '#FF0000', '#0000FF'])

                    print('finished') 

if __name__ == "__main__":
    datasets_names = [
                        #   'high-school-contacts',
                        #    'reality-call2',
                        #    'enron',
                           'dblp'
                    ]

    optins_methods = ['affinity']
    optins_perc = [.0]
    
    num_threads = 15

    es = [ .1 ]

    levels_length = 100000
    runs = 10

    exp = DPWeightedNets(datasets_names, optins_methods, optins_perc, levels_length, num_threads, es, runs)
    exp.find_point_of_decreasing()
    exp.pre_process()