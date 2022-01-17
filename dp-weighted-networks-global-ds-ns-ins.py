
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

                    mask_without_in_in = g.edges_without_in_in()
                    g_without_in_in = WGraph(G=gt.GraphView(g, efilt=mask_without_in_in), prune=True)

                    # mask_out_out = g.edges_out_out()
                    # g_out_out = WGraph(G=gt.GraphView(g, efilt=mask_out_out), prune=True)

                    len_all_edges_without_in_in = int( ( g.n() * (g.n()-1))/2 - (len(optins) * (len(optins)-1))/2 )

                    for e in self.es:
                        utils.log_msg('******* eps = ' + str(e) + ' *******')

                        # privacy budgets #
                        e1 = 0.3*e # budget for perturb edge weights
                        e2 = 0.3*e # budget for query node strength
                        e3 = 0.4*e # budget for query degree sequence

                        geom_prob_mass_e1 = dp_mechanisms.geom_prob_mass(e1)
                        geom_prob_mass_e2_1 = dp_mechanisms.geom_prob_mass(e2)
                        geom_prob_mass_e2_2 = dp_mechanisms.geom_prob_mass(e2, sensitivity=2)
                        geom_prob_mass_e3 = dp_mechanisms.geom_prob_mass(e3, sensitivity=2)

                        for r in range(self.runs):
                            utils.log_msg('....... RUN ' + str(r) + ' .......')   
                            
                            ds = g_without_in_in.degrees() # degree sequence
                            ds_noisy = dp_mechanisms.geometric(ds, geom_prob_mass_e3)
                            ds_ajusted = tools.min_l2_norm(ds_noisy, np.sum(ds_noisy), num_steps=10)
                            new_m = int(np.sum(ds_ajusted)/2)

                            edges_w = g_without_in_in.edges_w()
                            edges_w_noisy = dp_mechanisms.geometric(edges_w, geom_prob_mass_e1)
                            top_m_edges_w_noisy, num_remaining_edges, non_zero_edges_w_filtered_mask = tools.high_pass_filter(edges_w_noisy, e1, len_all_edges_without_in_in, new_m)

                            # G' is the graph only with original edges that pass to the filter
                            edges_g_prime = g_without_in_in.get_edges()[non_zero_edges_w_filtered_mask]
                            edges_w_prime = top_m_edges_w_noisy[:num_remaining_edges]
                            edges_in_g_prime = np.concatenate((edges_g_prime, np.array([edges_w_prime]).T ), axis=1)

                            g_prime = tools.build_g_from_edges(g, edges_in_g_prime, add_optin_edges=False)

                            ds_remaining = ds_ajusted - g_prime.degrees()
                            ds_remaining_adjusted = tools.min_l2_norm(ds_remaining, np.sum(ds_remaining), num_steps=10, min_value=0)

                            new_edges_before_ns_adjustment = tools.get_edges_from_degree_sequence(g_prime, ds_remaining_adjusted)  
                            new_edges_and_weights = np.concatenate((new_edges_before_ns_adjustment, np.array([top_m_edges_w_noisy[num_remaining_edges:len(top_m_edges_w_noisy)]]).T ), axis=1)
                            all_edges_before_ns_adjustment = np.append(edges_in_g_prime, new_edges_and_weights, axis=0) 

                            # G'' is the graph after degree sequence adjustment and before NS adjustment
                            g_prime2 = tools.build_g_from_edges(g, all_edges_before_ns_adjustment, add_optin_edges=False)

                            nss = g_without_in_in.node_strengths()
                            nss_noisy = np.zeros(n)
                            nss_noisy[optins] = dp_mechanisms.geometric(nss[optins], geom_prob_mass_e2_1)
                            nss_noisy[optouts] = dp_mechanisms.geometric(nss[optouts], geom_prob_mass_e2_2)
                            nss_ajusted = tools.min_l2_norm(nss_noisy, np.sum(nss_noisy), num_steps=10)

                            # degrees_out_out = g_out_out.degrees()[optouts]

                            new_edges = tools.adjust_edge_weights_based_on_ns(g_prime2, nss_ajusted)

                            # new_edges = np.concatenate((all_edges_before_ns_adjustment[:,[0,1]], np.array([edges_w_ajusted]).T ), axis=1)

                            # all_edges = np.append(edges_in_g_prime, new_edges, axis=0) 
                            new_g = tools.build_g_from_edges(g, new_edges)

                            # print("%.2f" % float(error_metrics.mre(egocentric_metrics.node_strength(g), egocentric_metrics.node_strength(new_g))))
                            # print("%.2f" % float(error_metrics.mre(g.degrees(), new_g.degrees())))
                            # print("%.2f" % float(error_metrics.mre(egocentric_metrics.sum_of_2_hop_edges(g), egocentric_metrics.sum_of_2_hop_edges(new_g)  )))

                            utils.log_msg('saving graph...')
                            path_graph = "./data/%s/exp/graph_perturbed_%s_ins%s_e%s_r%s_global_ds_ns_ins.graphml" % ( dataset , optin_method, optin_perc, e, r)     
                            new_g.save(path_graph)                            

if __name__ == "__main__":
    datasets_names = [
                    'high-school-contacts',
                    'copenhagen-interaction',
                    'reality-call',
                    'contacts-dublin'
                    # 'digg-reply'
                    #  'enron',
                    #  'wiki-talk',
                    # 'sx-stackoverflow'
                    ]

    optins_methods = ['affinity']
    optins_perc = [.2]

    es = [ .5 ]

    runs = 3

    exp = DPWeightedNets(datasets_names, optins_methods, optins_perc, es, runs)
    exp.run()