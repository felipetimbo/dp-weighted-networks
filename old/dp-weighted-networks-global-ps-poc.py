
import os
import numpy as np
import graph_tool as gt

from dpwnets import (utils, dp_mechanisms, tools, graphics)
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

                    len_all_edges_without_in_in = int( ( g.n() * (g.n()-1))/2 - (len(optins) * (len(optins)-1))/2 )

                    for e in self.es:
                        utils.log_msg('******* eps = ' + str(e) + ' *******')

                        geom_prob_mass_e = dp_mechanisms.geom_prob_mass(e) 

                        for r in range(self.runs):
                            utils.log_msg('....... RUN ' + str(r) + ' .......')   

                            edges_w = g_without_in_in.edges_w()
                            edges_w_noisy = dp_mechanisms.geometric(edges_w, geom_prob_mass_e)

                            utils.log_msg('high pass filter...')

                            top_m_edges_w_noisy, num_remaining_edges, non_zero_edges_w_filtered_mask = tools.high_pass_filter(edges_w_noisy, e, len_all_edges_without_in_in, g.m())

                            edges_g_prime = g_without_in_in.get_edges()[non_zero_edges_w_filtered_mask]
                            edges_w_prime = top_m_edges_w_noisy[:num_remaining_edges]
                            orig_edges_after_hpf = np.concatenate((edges_g_prime, np.array([edges_w_prime]).T ), axis=1)
                            
                            num_edges_to_be_created = len(top_m_edges_w_noisy) - num_remaining_edges
                            created_edges = tools.sample_random_edges(n, num_edges_to_be_created, set(map(tuple, edges_g_prime)), non_optins_pos)
                            created_edges_w = top_m_edges_w_noisy[num_remaining_edges:len(top_m_edges_w_noisy)]
                            created_edges_after_hpf = np.concatenate((created_edges, np.array([created_edges_w]).T ), axis=1)

                            all_edges_after_hpf = np.append(orig_edges_after_hpf, created_edges_after_hpf, axis=0)
                            g_hps = tools.build_g_from_edges(g, all_edges_after_hpf, add_optin_edges=False)
                            
                            utils.log_msg('threshold sampling...')

                            # top_m_edges_w_noisy, num_remaining_edges, non_zero_edges_w_filtered_mask = tools.high_pass_filter(edges_w_noisy, e1, len_all_edges_without_in_in, new_m)
                            top_m_edges_w_noisy, num_remaining_edges, non_zero_edges_w_filtered_mask = tools.priority_sampling(edges_w_noisy, e, len_all_edges_without_in_in, g.m())

                            edges_g_prime = g_without_in_in.get_edges()[non_zero_edges_w_filtered_mask]
                            edges_w_prime = top_m_edges_w_noisy[:num_remaining_edges]
                            orig_edges_after_hpf = np.concatenate((edges_g_prime, np.array([edges_w_prime]).T ), axis=1)
                            
                            num_edges_to_be_created = len(top_m_edges_w_noisy) - num_remaining_edges
                            created_edges = tools.sample_random_edges(n, num_edges_to_be_created, set(map(tuple, edges_g_prime)), non_optins_pos)
                            created_edges_w = top_m_edges_w_noisy[num_remaining_edges:len(top_m_edges_w_noisy)]
                            created_edges_after_ps = np.concatenate((created_edges, np.array([created_edges_w]).T ), axis=1)

                            all_edges_after_ps = np.append(orig_edges_after_hpf, created_edges_after_ps, axis=0)
                            
                            g_ps = tools.build_g_from_edges(g, all_edges_after_ps, add_optin_edges=False)

                            # utils.log_msg('plotting histograms...')
                            # path = "./data/%s/edges_w_histograms_%s_ins%s_e%s_r%s.png" % ( dataset , optin_method, optin_perc, e, r)     
                            # graphics.histogram3(g.ep.ew.fa.astype('int'), g_hps.ep.ew.fa.astype('int'), g_ps.ep.ew.fa.astype('int'), num_bins=200, path=path) 

                            all_edges_w_1 = all_edges_after_hpf[:,2]
                            all_edges_w_adjusted = tools.min_l2_norm_old(all_edges_w_1, np.sum(edges_w))
                            all_edges_hpf = np.concatenate((all_edges_after_hpf[:,[0,1]], np.array([all_edges_w_adjusted]).T ), axis=1)

                            g_hpf_2 = tools.build_g_from_edges(g, all_edges_hpf, add_optin_edges=False)

                            all_edges_w_2 = all_edges_after_ps[:,2]
                            all_edges_w_adjusted = tools.min_l2_norm_old(all_edges_w_2, np.sum(edges_w))
                            all_edges_ps = np.concatenate((all_edges_after_ps[:,[0,1]], np.array([all_edges_w_adjusted]).T ), axis=1)

                            g_ps_2 = tools.build_g_from_edges(g, all_edges_ps, add_optin_edges=False)

                            utils.log_msg('plotting histograms ADJUSTED...')
                            path = "./data/%s/edges_w_histograms_%s_ins%s_e%s_r%s_ADJUSTED.png" % ( dataset , optin_method, optin_perc, e, r)     
                            graphics.histogram3(g.ep.ew.fa.astype('int'), g_ps_2.ep.ew.fa.astype('int') , g_hpf_2.ep.ew.fa.astype('int') , max_x_value=200, num_bins=4000, path=path) 


if __name__ == "__main__":
    datasets_names = [
                        # 'high-school-contacts',
                        # 'copenhagen-interaction',
                        # 'reality-call',
                        # 'reality-call2',
                        # 'contacts-dublin',
                        # 'digg-reply',
                          'enron',
                        # 'wiki-talk',
                        #  'dblp'
                    ]

    optins_methods = ['affinity'] 
    optins_perc = [.0]

    es = [ 1 ]

    runs = 1

    exp = DPWeightedNets(datasets_names, optins_methods, optins_perc, es, runs)
    exp.run()