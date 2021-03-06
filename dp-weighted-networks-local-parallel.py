
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
                    optouts = g.optouts()
                    len_optouts = len(optouts)
                    # all_ns = gt.incident_edges_op(ego_graph, "out", "sum", ego_graph.ep.ew)

                    mask_in_out = g.edges_in_out()
                    g_in_out = WGraph(G=gt.GraphView(g, efilt=mask_in_out), prune=True)

                    mask_out_out = g.edges_out_out()
                    g_out_out = WGraph(G=gt.GraphView(g, efilt=mask_out_out), prune=True)

                    for e in self.es:
                        utils.log_msg('******* eps = ' + str(e) + ' *******')

                        # privacy budgets #
                        e1 = 0.3*e # budget for perturb edge weights
                        e2 = 0.3*e # budget for query node strength
                        e3 = 0.4*e # budget for query degree 

                        geom_prob_mass_e1 = dp_mechanisms.geom_prob_mass(e1)
                        geom_prob_mass_e2 = dp_mechanisms.geom_prob_mass(e2)
                        geom_prob_mass_e3_s1 = dp_mechanisms.geom_prob_mass(e3)
                        geom_prob_mass_e3_s2 = dp_mechanisms.geom_prob_mass(e3, sensitivity=2) # quering opt-out edges has sens = 2

                        for r in range(self.runs):
                            utils.log_msg('....... RUN ' + str(r) + ' .......')   
                            
                            new_edges = np.empty((0,3), int)
                            strengths_noisy = np.array([])
                            degrees_noisy = np.array([])

                            ### local processing ###
                            range_v = np.array_split(list(range(g.num_vertices())), self.num_threads) 
                            utils.log_msg('running local DP...')

                            # with Manager() as manager:
                            manager = multiprocessing.Manager()
                            new_edges_multiprocessing = manager.list() 
                            strengths_noisy_multiprocessing = manager.list() 
                            degrees_noisy_multiprocessing = manager.list() 

                            threads = []
                            for i in range(self.num_threads):
                                t = multiprocessing.Process( target = self.local_dp, args =(new_edges_multiprocessing, strengths_noisy_multiprocessing, 
                                                                degrees_noisy_multiprocessing, optins_mask, range_v[i], g_in_out, optouts, g_out_out, 
                                                                geom_prob_mass_e3_s2, geom_prob_mass_e2, geom_prob_mass_e1, e1, len_optouts))
                                t.start()
                                threads.append(t)
                            
                            for t in threads:    
                                t.join()

                            ### aggregator ###

                            new_edges = np.append(new_edges, np.concatenate( np.array(list(new_edges_multiprocessing),dtype='object')), axis=0).astype('int')
                            strengths_noisy = np.array(list(strengths_noisy_multiprocessing), dtype='int') # np.append(strengths_noisy, np.concatenate( np.array(list(strengths_noisy_multiprocessing),dtype='object')), axis=0).astype('int')
                            degrees_noisy = np.array(list(degrees_noisy_multiprocessing), dtype='int') # np.append(degrees_noisy, np.concatenate( np.array(list(degrees_noisy_multiprocessing),dtype='object')), axis=0).astype('int')

                            utils.log_msg('post processing graph...')
                            new_edges = tools.mean_of_duplicated_edges(new_edges)
                            g_before_pp = tools.build_g_from_edges(g, new_edges)

                            # num_edges_opt_outs = int(np.sum( np.array(degrees_noisy)[optouts])/2)
                            sum_edges_opt_outs = int(np.sum( np.array(strengths_noisy)[optouts])/2)
                            
                            # num_edges_opt_outs = int(g.m() - np.sum(g_before_pp.edges_in_out()) - np.sum(g_before_pp.edges_in_in()))
                            num_edges_opt_outs = g_out_out.m()
                            new_g = tools.remove_edges_with_lower_weights_and_adjust(g_before_pp, num_edges_opt_outs , sum_edges_opt_outs)
                            
                            utils.log_msg('saving graph...')
                            path_graph = "./data/%s/exp/graph_perturbed_%s_ins%s_e%s_r%s_local_p3.graphml" % ( dataset , optin_method, optin_perc, e, r)     
                            new_g.save(path_graph)                            

    def local_dp(self, new_edges, strengths_noisy, degrees_noisy, optins_mask, range_v, g_in_out, optouts, g_out_out, geom_prob_mass_e3_s2, geom_prob_mass_e2, geom_prob_mass_e1, e1, len_optouts):
        
        for v in range_v:
            if optins_mask[v]:
                ego_graph = g_in_out
                optouts_minus_self_edge = optouts
            else:
                ego_graph = g_out_out
                optouts_minus_self_edge = optouts[optouts != v]
            
            d = ego_graph.vertex(v).out_degree()
            d_noisy = dp_mechanisms.geometric([d], geom_prob_mass_e3_s2)[0]
            degrees_noisy.append(d_noisy) 
            # degrees_noisy = np.append(degrees_noisy, d_noisy, axis=0)
            
            neighbors_v = ego_graph.get_out_edges(v, [ego_graph.ep.ew] )
            ns = np.sum(neighbors_v[:,2])
            edges_w_sum_noisy = dp_mechanisms.geometric([ns], geom_prob_mass_e2)[0]
            strengths_noisy.append(edges_w_sum_noisy)
            # strengths_noisy = np.append(strengths_noisy, edges_w_sum_noisy, axis=0)  
            
            if d_noisy > 0 and d_noisy < ego_graph.max_degree():
            
                edges_w = neighbors_v[:,2] 
                edges_w_noisy = dp_mechanisms.geometric(edges_w, geom_prob_mass_e1)
                top_d_edges_w_noisy, num_remaining_edges, non_zero_edges_w_filtered_mask = tools.high_pass_filter(edges_w_noisy, e1, len_optouts, d_noisy)                                   
            
                edges_w_ajusted = tools.min_l2_norm(top_d_edges_w_noisy, edges_w_sum_noisy, num_steps=10)
            
                edges_w_noisy_non_zero = edges_w_ajusted[:num_remaining_edges]
                edges_w_noisy_zeros = edges_w_ajusted[num_remaining_edges:len(edges_w_ajusted)]    
            
                opt_outs_picked = np.random.choice(optouts_minus_self_edge, len(edges_w_noisy_zeros), replace=False)
            
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
                    #  'copenhagen-interaction'
                    # 'reality-call'
                    #  'contacts-dublin'
                    #    'digg-reply', 
                    #    'enron' 
                    # 'wiki-talk'
                    ]

    optins_methods = ['affinity']
    optins_perc = [.2]

    es = [ .1, 1, 2 ]

    runs = 2
    num_threads = 6

    exp = DPWeightedNets(datasets_names, optins_methods, optins_perc, es, num_threads, runs)
    exp.run()