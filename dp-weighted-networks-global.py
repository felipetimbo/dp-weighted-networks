
import os
import math
import numpy as np
import graph_tool as gt

from dpwnets import utils
from dpwnets import dp_mechanisms
from dpwnets import tools

from graph.wgraph import WGraph

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

                    optins = g.optins()
                    # optouts = g.optouts()

                    mask_without_in_in = g.edges_without_in_in()
                    g_without_in_in = WGraph(G=gt.GraphView(g, efilt=mask_without_in_in), prune=True)

                    len_all_edges_without_in_in = int( ( g.n() * (g.n()-1))/2 - (len(optins) * (len(optins)-1))/2 )

                    for e in self.es:
                        utils.log_msg('******* eps = ' + str(e) + ' *******')

                        # privacy budgets #
                        e1 = 0.9*e # budget for perturb edges
                        e2 = 0.1*e # budget for query total sum of edge weights

                        geom_prob_mass_e1 = dp_mechanisms.geom_prob_mass(e1)
                        geom_prob_mass_e2 = dp_mechanisms.geom_prob_mass(e2)

                        for r in range(self.runs):
                            utils.log_msg('....... RUN ' + str(r) + ' .......')   
                            
                            edges_w = g_without_in_in.edges_w()
                            edges_w_noisy = dp_mechanisms.geometric(edges_w, geom_prob_mass_e1)
                            top_m_edges_w_noisy, num_remaining_edges, non_zero_edges_w_filtered_mask = tools.high_pass_filter(edges_w_noisy, e1, len_all_edges_without_in_in, g_without_in_in.m())

                            edges_w_sum = np.sum(edges_w)
                            edges_w_sum_noisy = dp_mechanisms.geometric([edges_w_sum], geom_prob_mass_e2)[0]
                            edges_w_ajusted = tools.min_l2_norm(top_m_edges_w_noisy, edges_w_sum_noisy, num_steps=10)

                            edges_w_noisy_non_zero = np.zeros(len(edges_w_noisy))
                            edges_w_noisy_non_zero[non_zero_edges_w_filtered_mask] = edges_w_ajusted[:num_remaining_edges]
                            edges_w_noisy_zeros = edges_w_ajusted[num_remaining_edges:len(edges_w_ajusted)]

                            utils.log_msg('sampling graph...')
                            new_g = tools.sample_graph(g, edges_w_noisy_non_zero, edges_w_noisy_zeros)

                            utils.log_msg('saving graph...')
                            path_graph = "./data/%s/exp/graph_perturbed_%s_ins%s_e%s_r%s_baseline.graphml" % ( dataset , optin_method, optin_perc, e, r)     
                            new_g.save(path_graph)                            

if __name__ == "__main__":
    datasets_names = [
                    'high-school-contacts',
                    'copenhagen-interaction',
                    'reality-call', 
                    'contacts-dublin',
                    'digg-reply']
                    # 'wiki-talk',
                    # 'sx-stackoverflow']

    optins_methods = ['affinity']
    optins_perc = [.2]

    es = [ .1, 1, 2 ]

    runs = 5

    exp = DPWeightedNets(datasets_names, optins_methods, optins_perc, es, runs)
    exp.run()