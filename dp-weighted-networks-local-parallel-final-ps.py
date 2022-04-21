import os
import math
import numpy as np
import graph_tool.all as gt
import threading
import multiprocessing

from dpwnets import utils
from dpwnets import dp_mechanisms
from dpwnets import tools

from graph.wgraph import WGraph

np.random.seed(0)

class DPWeightedNets():

    def __init__(self, 
                    datasets_names, 
                    optins_methods, 
                    optins_perc, 
                    es, 
                    num_threads,
                    runs
                ):
        self.datasets_names = datasets_names  
        self.optins_methods = optins_methods
        self.optins_perc = optins_perc
        self.es = es
        self.num_threads = num_threads
        self.runs = runs        

    def run(self):
        for dataset in self.datasets_names:
            utils.log_msg('*************** DATASET = ' + dataset + ' ***************')

            for optin_method in self.optins_methods: 
                for optin_perc in self.optins_perc:
                    url = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, '%s_%s_%s.graphml' % (dataset, optin_method, optin_perc )))
                    g = WGraph(url)

                    optins_mask = g.vp.optin.fa.astype(bool)
                    optins = g.optins()
                    optouts = g.optouts()
                    len_optouts = len(optouts)
                    non_optins_pos = g.vp.optin.fa == 0
                    # all_ns = gt.incident_edges_op(ego_graph, "out", "sum", ego_graph.ep.ew)

                    # mask_in_out = g.edges_in_out() 
                    # g_in_out = WGraph(G=gt.GraphView(g, efilt=mask_in_out), prune=True) 

                    # mask_out_out = g.edges_out_out()
                    # g_out_out = WGraph(G=gt.GraphView(g, efilt=mask_out_out), prune=True)

                    mask_without_in_in = g.edges_without_in_in()
                    g_without_in_in = WGraph(G=gt.GraphView(g, efilt=mask_without_in_in), prune=True)

                    orig_edges_w = g_without_in_in.edges_w()

                    for e in self.es:
                        utils.log_msg('******* eps = ' + str(e) + ' *******')

                        # privacy budgets #
                        e1 = 0.5*e # budget for perturb edge weights
                        # e2 = 0.3*e # budget for query node strength
                        e3 = 0.5*e # budget for query degree 

                        geom_prob_mass_e1 = dp_mechanisms.geom_prob_mass(e1)
                        # geom_prob_mass_e2 = dp_mechanisms.geom_prob_mass(e2)
                        # geom_prob_mass_e3_s1 = dp_mechanisms.geom_prob_mass(e3)
                        geom_prob_mass_e3_s2 = dp_mechanisms.geom_prob_mass(e3, sensitivity=2) # quering opt-out edges has sens = 2

                        for r in range(self.runs):
                            utils.log_msg('....... RUN ' + str(r) + ' .......')   
                            
                            new_edges = np.empty((0,3), int)
                            # strengths_noisy = np.array([])
                            # degrees_noisy = np.array([])

                            ### local processing ###
                            range_v = np.array_split(list(range(g.num_vertices())), self.num_threads) 
                            utils.log_msg('running local DP...')

                            # with Manager() as manager:
                            manager = multiprocessing.Manager()
                            new_edges_multiprocessing = manager.list() 
                            # strengths_noisy_multiprocessing = manager.dict() 
                            degrees_noisy_multiprocessing = manager.dict() 

                            threads = []
                            for i in range(self.num_threads):
                                t = multiprocessing.Process( target = self.local_dp, args =(new_edges_multiprocessing,  
                                                                degrees_noisy_multiprocessing,optins_mask, range_v[i],g_without_in_in, optins, optouts, 
                                                                geom_prob_mass_e3_s2, geom_prob_mass_e1, e1, len_optouts))
                                t.start()
                                threads.append(t)
                            
                            for t in threads:    
                                t.join()

                            ### aggregator ###

                            new_edges = np.append(new_edges, np.concatenate( np.array(list(new_edges_multiprocessing),dtype='object')), axis=0).astype('int')
                            # strengths_noisy_dict = dict(strengths_noisy_multiprocessing) #, dtype='int') # np.append(strengths_noisy, np.concatenate( np.array(list(strengths_noisy_multiprocessing),dtype='object')), axis=0).astype('int')
                            # strengths_noisy = np.array([strengths_noisy_dict[key] for key in sorted(strengths_noisy_dict.keys())])

                            degrees_noisy_dict = dict(degrees_noisy_multiprocessing) #, dtype='int') # np.append(degrees_noisy, np.concatenate( np.array(list(degrees_noisy_multiprocessing),dtype='object')), axis=0).astype('int')
                            degrees_noisy = np.array([degrees_noisy_dict[key] for key in sorted(degrees_noisy_dict.keys())])
                            ds_remaining_adjusted = tools.min_l2_norm_old(degrees_noisy, np.sum(degrees_noisy), num_steps=10, min_value=1)
                            new_m = int(np.sum(ds_remaining_adjusted)/2)

                            utils.log_msg('merging graph...')
                            new_edges_after_mean = tools.mean_of_all_edges(new_edges)
                            top_m_edges = tools.top_m_edges_with_lower_weights(new_edges_after_mean, new_m, g)

                            all_edges_w = top_m_edges[:,2]
                            all_edges_w_adjusted = tools.min_l2_norm_old(all_edges_w, np.sum(orig_edges_w))
                            all_edges = np.concatenate((top_m_edges[:,[0,1]], np.array([all_edges_w_adjusted]).T ), axis=1)
                            
                            g_priority_sampled = tools.build_g_from_edges(g, all_edges, add_optin_edges=False)

                            # g_before_pp = tools.build_g_from_edges(g, top_m_edges, add_optin_edges=False)
                            
                            # num_edges_to_remain = int(np.sum(degrees_noisy)/2) 
                            # new_edges_after_capping = tools.remove_edges_with_lower_weights(g_before_pp, num_edges_to_remain )
                            # g_before_degree_adjustment = tools.build_g_from_edges(g, new_edges_after_capping, add_optin_edges=False)

                            utils.log_msg('adjusting degrees ...')

                            edges_with_deg_seq_adjusted = tools.adjust_degree_sequence(g_priority_sampled, ds_remaining_adjusted, non_optins_pos)
                            new_g = tools.build_g_from_edges(g, edges_with_deg_seq_adjusted, add_optin_edges=False)

                            # degrees_difference = (g_before_degree_adjustment.degrees() - ds_remaining_adjusted).astype(int)
                            # edges_before_ns_adjustment = tools.adjust_degree_sequence(g_before_pp, ds_remaining_adjusted, non_optins_pos )
                            
                            # utils.log_msg('node strength adjustment...')
                            # g_prime2 = tools.build_g_from_edges(g, edges_before_ns_adjustment, add_optin_edges=False)
                            # nss_ajusted = tools.min_l2_norm_old(strengths_noisy, np.sum(strengths_noisy), num_steps=10)

                            # new_edges = tools.adjust_edge_weights_based_on_ns(g_prime2, nss_ajusted)
                            # new_g = tools.build_g_from_edges(g, new_edges)

                            utils.log_msg('saving graph...')
                            path_graph = "./data/%s/exp/graph_perturbed_%s_ins%s_e%s_r%s_local_final_ps.graphml" % ( dataset , optin_method, optin_perc, e, r)     
                            new_g.save(path_graph)                            

    def local_dp(self, new_edges, degrees_noisy, optins_mask, range_v, g_without_in_in, optins, optouts, geom_prob_mass_e3_s2, geom_prob_mass_e1, e1, len_optouts):
        
        for v in range_v:
            if optins_mask[v]:
                # ego_graph = g_in_out
                possible_edges = optouts
            else:
                # ego_graph = g_out_out
                possible_edges = np.append( optouts[optouts != v] , optins )
            
            d = g_without_in_in.vertex(v).out_degree()
            d_noisy = dp_mechanisms.geometric2([d], geom_prob_mass_e3_s2)[0]
            degrees_noisy[v] = d_noisy 
            # degrees_noisy.append(d_noisy) 
            # degrees_noisy = np.append(degrees_noisy, d_noisy, axis=0)
            
            neighbors_v = g_without_in_in.get_out_edges(v, [g_without_in_in.ep.ew] )
            
            # ns = np.sum(neighbors_v[:,2])
            # edges_w_sum_noisy = dp_mechanisms.geometric2([ns], geom_prob_mass_e2)[0]
            # strengths_noisy[v] = edges_w_sum_noisy
            # strengths_noisy.append(edges_w_sum_noisy)
            # strengths_noisy = np.append(strengths_noisy, edges_w_sum_noisy, axis=0)  
            
            if d_noisy > 0 and d_noisy < g_without_in_in.max_degree():
            
                edges_w = neighbors_v[:,2] 
                edges_w_noisy = dp_mechanisms.geometric(edges_w, geom_prob_mass_e1)
                # top_d_edges_w_noisy, num_remaining_edges, non_zero_edges_w_filtered_mask = tools.high_pass_filter(edges_w_noisy, e1, len_optouts, d_noisy)                                   
                top_d_edges_w_noisy, num_remaining_edges, non_zero_edges_w_filtered_mask = tools.priority_sampling(edges_w_noisy, e1, len_optouts, d_noisy)
            
                edges_w_ajusted = tools.min_l2_norm_old(top_d_edges_w_noisy, np.sum(edges_w), num_steps=10)
            
                edges_w_noisy_non_zero = edges_w_ajusted[:num_remaining_edges]
                edges_w_noisy_zeros = edges_w_ajusted[num_remaining_edges:len(edges_w_ajusted)]    
            
                opt_outs_picked = np.random.choice(possible_edges, len(edges_w_noisy_zeros), replace=False)
            
                new_edge_zeros = np.column_stack((np.ones(len(edges_w_noisy_zeros))*v, opt_outs_picked ))
                new_edge_zeros = np.concatenate((new_edge_zeros, np.array([edges_w_noisy_zeros]).T ), axis=1)
                if len(new_edge_zeros) > 0:
                    new_edges.append(new_edge_zeros)  
            
                new_edge_non_zeros = np.concatenate((neighbors_v[:,[0,1]][non_zero_edges_w_filtered_mask], np.array([edges_w_noisy_non_zero]).T ), axis=1)
                if len(new_edge_non_zeros) > 0:
                    new_edges.append(new_edge_non_zeros)  
        
        utils.log_msg('thread finished' )

if __name__ == "__main__":
    datasets_names = [
                        #  'high-school-contacts',
                        # 'copenhagen-interaction',
                        #   'reality-call2',
                        # # 'contacts-dublin',
                        # # 'digg-reply', 
                        #   'enron' ,
                        # # 'wiki-talk',
                           'dblp'
                    ]

    optins_methods = ['affinity']
    optins_perc = [.0]

    es = [ .5, 2 ] 

    runs = 1
    num_threads = 15

    exp = DPWeightedNets(datasets_names, optins_methods, optins_perc, es, num_threads, runs)
    exp.run()