
import os
import math
import numpy as np
import graph_tool as gt
import mpmath

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

                    # mask_out_out = g.edges_out_out()
                    # g_out_out = WGraph(G=gt.GraphView(g, efilt=mask_out_out), prune=True)

                    # len_all_edges_without_in_in = int( ( g.n() * (g.n()-1))/2 - (len(optins) * (len(optins)-1))/2 )

                    edges_w = g_without_in_in.edges_w()
                    num_total_edges = int(g_without_in_in.n()*(g_without_in_in.n()-1)/2)
                    all_edges_w = np.append(edges_w, [0] * (num_total_edges - len(edges_w)))

                    all_edges_w = np.sort( np.random.choice(all_edges_w, 100, replace=False) )[::-1]

                    utils.log_msg('computing levels...')
                    levels_size = dp_mechanisms.get_levels_size(all_edges_w)

                    picked_levels = []
                    normalized_prob_masses = []
                    graphic_pos = np.linspace(0, len(levels_size)-2, num=1000, dtype=int)

                    for e in self.es:
                        utils.log_msg('******* eps = ' + str(e) + ' *******')

                        for r in range(self.runs):
                            utils.log_msg('....... RUN ' + str(r) + ' .......')   
                            
                            utils.log_msg('computing probability mass...')
                            prob_mass = dp_mechanisms.compute_probability_mass(levels_size, e)

                            utils.log_msg('running exponential...')
                            picked_level, normalized_prob_mass = dp_mechanisms.exponential_mechanism(prob_mass)

                            normalized_prob_masses.append( np.array(normalized_prob_mass)[graphic_pos])
                            picked_levels.append(picked_level)
                            
                    path_graphic = "./data/%s/levels/prob_mass_only_up.png" % ( dataset )

                    legends = [
                                '$\epsilon$=0.1',
                                '$\epsilon$=0.5',
                                '$\epsilon$=1.0'
                            ]

                    # graphics.line_plot2( np.array(list(range(1, len(levels_size))))[graphic_pos], normalized_prob_masses,
                    #                         xlabel='level', ylabel= 'prob. mass', ylog=True, # ylim=(min(normalized_prob_mass), None),
                    #                         path=path_graphic, line_legends=legends, figsize=(10, 5))

                    graphics.line_plot2( np.array(list(range(1, len(levels_size))))[graphic_pos], normalized_prob_masses,
                                            xlabel='level', ylabel= 'prob. mass', ylog=True, xlim=(0, 25000),
                                            path=path_graphic, line_legends=legends, figsize=(10, 5), colors = ['#000000', '#FF0000', '#0000FF'])

                    print(picked_levels) 


                            # group_output
                            # edges_w_noisy = dp_mechanisms.geometric(edges_w, geom_prob_mass_e)
                            # top_m_edges_w_noisy, num_remaining_edges, non_zero_edges_w_filtered_mask = tools.high_pass_filter(edges_w_noisy, e, len_all_edges_without_in_in, new_m)

                            # edges_g_prime = g_without_in_in.get_edges()[non_zero_edges_w_filtered_mask]
                            # edges_w_prime = top_m_edges_w_noisy[:num_remaining_edges]
                            # orig_edges_after_hpf = np.concatenate((edges_g_prime, np.array([edges_w_prime]).T ), axis=1)
                            
                            # num_edges_to_be_created = len(top_m_edges_w_noisy) - num_remaining_edges
                            # created_edges = tools.sample_random_edges(n, num_edges_to_be_created, set(map(tuple, edges_g_prime)), non_optins_pos)
                            # created_edges_w = top_m_edges_w_noisy[num_remaining_edges:len(top_m_edges_w_noisy)]
                            # created_edges_after_ts = np.concatenate((created_edges, np.array([created_edges_w]).T ), axis=1)

                            # all_edges_after_ts = np.append(orig_edges_after_hpf, created_edges_after_ts, axis=0)
                            # g_sampled = tools.build_g_from_edges(g, all_edges_after_ts, add_optin_edges=False)

                            # utils.log_msg('saving baseline ...')
                            # path_graph = "./data/%s/exp/%s_ins%s_e%s_r%s_baseline5.graphml" % ( dataset , optin_method, optin_perc, e, r)     
                            # g_sampled.save(path_graph)  

                            
if __name__ == "__main__":
    datasets_names = [
                          'high-school-contacts',
                        #   'reality-call2',
                        #   'enron',
                        #   'dblp'
                    ]

    optins_methods = ['affinity']
    optins_perc = [.0]

    es = [ .1, .5, 1 ]

    runs = 1

    exp = DPWeightedNets(datasets_names, optins_methods, optins_perc, es, runs)
    exp.run()