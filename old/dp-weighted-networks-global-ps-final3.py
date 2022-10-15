
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

                    # mask_out_out = g.edges_out_out()
                    # g_out_out = WGraph(G=gt.GraphView(g, efilt=mask_out_out), prune=True)

                    len_all_edges_without_in_in = int( ( g.n() * (g.n()-1))/2 - (len(optins) * (len(optins)-1))/2 )

                    for e in self.es:
                        utils.log_msg('******* eps = ' + str(e) + ' *******')

                        # privacy budgets #
                        e1 = 0.5*e # budget for perturb edge weights
                        e2 = 0.5*e # budget for query degree sequence

                        geom_prob_mass_e1 = dp_mechanisms.geom_prob_mass(e1)
                        geom_prob_mass_e2 = dp_mechanisms.geom_prob_mass(e2, sensitivity=2)

                        for r in range(self.runs):
                            utils.log_msg('....... RUN ' + str(r) + ' .......')   
                            
                            ds = g_without_in_in.degrees() # degree sequence 
                            ds_noisy = dp_mechanisms.geometric(ds, geom_prob_mass_e2)
                            ds_ajusted = tools.min_l2_norm_old(ds_noisy, np.sum(ds_noisy), num_steps=10)
                            new_m = int(np.sum(ds_ajusted)/2) 

                            utils.log_msg('threshold sampling...')

                            edges_w = g_without_in_in.edges_w()
                            edges_w_noisy = dp_mechanisms.geometric(edges_w, geom_prob_mass_e1)
                            # top_m_edges_w_noisy, num_remaining_edges, non_zero_edges_w_filtered_mask = tools.high_pass_filter(edges_w_noisy, e1, len_all_edges_without_in_in, new_m)
                            top_m_edges_w_noisy, num_remaining_edges, non_zero_edges_w_filtered_mask = tools.priority_sampling(edges_w_noisy, e1, len_all_edges_without_in_in, new_m)

                            edges_g_prime = g_without_in_in.get_edges()[non_zero_edges_w_filtered_mask]
                            edges_w_prime = top_m_edges_w_noisy[:num_remaining_edges]
                            orig_edges_after_hpf = np.concatenate((edges_g_prime, np.array([edges_w_prime]).T ), axis=1)
                            
                            num_edges_to_be_created = len(top_m_edges_w_noisy) - num_remaining_edges
                            created_edges = tools.sample_random_edges(n, num_edges_to_be_created, set(map(tuple, edges_g_prime)), non_optins_pos)
                            created_edges_w = top_m_edges_w_noisy[num_remaining_edges:len(top_m_edges_w_noisy)]
                            created_edges_after_ps = np.concatenate((created_edges, np.array([created_edges_w]).T ), axis=1)

                            all_edges_after_ps = np.append(orig_edges_after_hpf, created_edges_after_ps, axis=0)
                            
                            all_edges_w = all_edges_after_ps[:,2]
                            all_edges_w_adjusted = tools.min_l2_norm_old(all_edges_w, np.sum(edges_w))
                            all_edges = np.concatenate((all_edges_after_ps[:,[0,1]], np.array([all_edges_w_adjusted]).T ), axis=1)
                            
                            g_priority_sampled = tools.build_g_from_edges(g, all_edges, add_optin_edges=False)

                            # graphics.histogram(g.ep.ew.fa.astype('int'), g_priority_sampled.ep.ew.fa.astype('int'), path="./data/%s/comparison.png" % dataset  )
                            
                            # path_graph_baseline = "./data/%s/exp/graph_perturbed_%s_ins%s_e%s_r%s_ps2_baseline.graphml" % ( dataset , optin_method, optin_perc, e, r)     
                            # g_priority_sampled.save(path_graph_baseline)    

                            utils.log_msg('adjusting degrees ...')

                            edges_with_deg_seq_adjusted = tools.adjust_degree_sequence(g_priority_sampled, ds_ajusted, non_optins_pos)
                            new_g = tools.build_g_from_edges(g, edges_with_deg_seq_adjusted, add_optin_edges=False)

                            utils.log_msg('saving graph...')
                            path_graph = "./data/%s/exp/graph_perturbed_%s_ins%s_e%s_r%s_global_ps2.graphml" % ( dataset , optin_method, optin_perc, e, r)     
                            new_g.save(path_graph)                            

if __name__ == "__main__":
    datasets_names = [
                        # 'high-school-contacts',
                        # 'copenhagen-interaction',
                        # 'reality-call',
                        # 'contacts-dublin',
                        # 'digg-reply',
                        # 'enron',
                        # 'wiki-talk',
                       'dblp'
                    ]

    optins_methods = ['affinity'] 
    optins_perc = [.0]

    es = [ 1  ]

    runs = 5

    exp = DPWeightedNets(datasets_names, optins_methods, optins_perc, es, runs)
    exp.run()