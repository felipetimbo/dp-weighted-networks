
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

                        # privacy budgets #
                        # e1 = 0.45*e # budget for perturb edge weights 
                        # e2 = 0.1*e # budget for query all node strengths 
                        # e3 = 0.45*e # budget for query degree sequence 

                        # geom_prob_mass_e = dp_mechanisms.geom_prob_mass(e, sensitivity=max(edges[:,2]))
                        # geom_prob_mass_e2_1 = dp_mechanisms.geom_prob_mass(e2) 
                        # geom_prob_mass_e2_2 = dp_mechanisms.geom_prob_mass(e2, sensitivity=2)
                        # geom_prob_mass_e3 = dp_mechanisms.geom_prob_mass(e3, sensitivity=2)

                        for r in range(self.runs):
                            utils.log_msg('....... RUN ' + str(r) + ' .......')   
                            
                            # ds = g_without_in_in.degrees() # degree sequence
                            # ds_noisy = dp_mechanisms.geometric(ds, geom_prob_mass_e3)
                            # ds_ajusted = tools.min_l2_norm_old(ds_noisy, np.sum(ds_noisy), num_steps=10)
                            # new_m = int(np.sum(ds_ajusted)/2)

                            # utils.log_msg('high pass filter...') 

                            edges_w_noisy = dp_mechanisms.geometric_mechanism(edges[:,2], e, sensitivity) 
                            edges_w_noisy_clipped = np.clip(edges_w_noisy, 1, None) 
                            new_edges = np.concatenate((edges[:,[0,1]], np.array([edges_w_noisy_clipped]).T ), axis=1)

                            new_g = tools.build_g_from_edges(g, new_edges)

                            # utils.log_msg('adjusting node strengths locally ...')

                            # final_edges = tools.adjust_edge_weights_based_on_ns(g_edges_globally_adj, nss_ajusted)
                            # final_edges = tools.min_l2_norm3(g_degrees_adjusted, nss_ajusted)

                            # new_g = tools.build_g_from_edges(g, final_edges)

                            utils.log_msg('saving graph...')
                            path_graph = "./data/%s/exp/%s_ins%s_e%s_r%s_baseline1.graphml" % ( dataset , optin_method, optin_perc, e, r)     
                            new_g.save(path_graph)                            

if __name__ == "__main__":
    datasets_names = [
                    #    'high-school-contacts',
                    #  'copenhagen-interaction',
                        'reality-call2',
                    # 'contacts-dublin',
                    # 'digg-reply',
                    #    'enron',
                    # 'wiki-talk'
                    # 'sx-stackoverflow'
                        # 'dblp'
                    ]

    optins_methods = ['affinity']
    optins_perc = [.0]

    es = [ .1, .5, 1 ]

    runs = 5

    exp = DPWeightedNets(datasets_names, optins_methods, optins_perc, es, runs)
    exp.run()