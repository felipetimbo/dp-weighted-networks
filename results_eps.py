import os
import numpy as np

from dpwnets import (utils, graphics)
from metrics import (error_metrics, egocentric_metrics)

from graph.wgraph import WGraph

class ResultsDPWeightedNets():

    def __init__(self, 
                    datasets_names, 
                    optins_methods, 
                    optins_perc, 
                    es, 
                    error_met, 
                    ego_metrics,
                    runs
                ):
        self.datasets_names = datasets_names  
        self.optins_methods = optins_methods
        self.optins_perc = optins_perc
        self.es = es
        self.error_met = error_met
        self.ego_metrics = ego_metrics
        self.runs = runs 

    def run(self):
        
        for dataset in self.datasets_names:
            utils.log_msg('*************** DATASET = ' + dataset + ' ***************')
            for optin_method in self.optins_methods: 
                for optin_perc in self.optins_perc:
                    url = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, '%s_%s_%s.graphml' % (dataset, optin_method, optin_perc )))
                    g = WGraph(url)

                    ego_metrics_true = {}

                    errors_1 = {} # approach 1
                    errors_2 = {} # approach 2
                    errors_3 = {} # approach 3

                    for ego_metric in self.ego_metrics:
                        ego_metrics_true[ego_metric] = egocentric_metrics.calculate(g, ego_metric )
                        errors_1[ego_metric] = {}
                        errors_2[ego_metric] = {}
                        errors_3[ego_metric] = {}

                        for error_metr in self.error_met: 
                            errors_1[ego_metric][error_metr] = []
                            errors_2[ego_metric][error_metr] = []
                            errors_3[ego_metric][error_metr] = []

                    for e in self.es:
                        utils.log_msg('*************** e = ' + str(e) + ' ***************')
                    
                        errors_list_1 = {} # approach 1
                        errors_list_2 = {} # approach 2
                        errors_list_3 = {} # approach 3

                        for ego_metric in self.ego_metrics:
                            errors_list_1[ego_metric] = {}
                            errors_list_2[ego_metric] = {}
                            errors_list_3[ego_metric] = {}

                            for error_metr in self.error_met: 
                                errors_list_1[ego_metric][error_metr] = []
                                errors_list_2[ego_metric][error_metr] = []
                                errors_list_3[ego_metric][error_metr] = []
                                   
                        for r in range(self.runs):    
                            path_g1 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, 'exp', 'graph_perturbed_%s_ins%s_e%s_r%s_global.graphml' % ( optin_method, optin_perc, e, r )))
                            g1 = WGraph(path_g1)

                            path_g2 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, 'exp', 'graph_perturbed_%s_ins%s_e%s_r%s_global_ds.graphml' % ( optin_method, optin_perc, e, r )))
                            g2 = WGraph(path_g2)

                            path_g3 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, 'exp', 'graph_perturbed_%s_ins%s_e%s_r%s_local.graphml' % ( optin_method, optin_perc, e, r )))
                            g3 = WGraph(path_g3)

                            for ego_metr in self.ego_metrics:
                                ego_metric_pred_1 = egocentric_metrics.calculate(g1, ego_metr)
                                ego_metric_pred_2 = egocentric_metrics.calculate(g2, ego_metr)
                                ego_metric_pred_3 = egocentric_metrics.calculate(g3, ego_metr)
                            
                                for error_metr in self.error_met: 
                                    if ego_metr == 'edges_w':
                                            
                                        error_1 = error_metrics.calculate_error_edges_w( error_metr, ego_metrics_true[ego_metr], ego_metric_pred_1)                   
                                        errors_list_1[ego_metr][error_metr].append(error_1)
                                        utils.log_msg('g1 global %s %s = %s' % ( error_metr, ego_metr, error_1 ) )

                                        error_2 = error_metrics.calculate_error_edges_w( error_metr, ego_metrics_true[ego_metr], ego_metric_pred_2)                   
                                        errors_list_2[ego_metr][error_metr].append(error_2)
                                        utils.log_msg('g2 global ds %s %s = %s' % ( error_metr, ego_metr, error_2 ) )

                                        error_3 = error_metrics.calculate_error_edges_w( error_metr, ego_metrics_true[ego_metr], ego_metric_pred_3)                   
                                        errors_list_3[ego_metr][error_metr].append(error_3)
                                        utils.log_msg('g3 local %s %s = %s' % ( error_metr, ego_metr, error_3 ) )
                                    
                                    else:
                                        error_1 = error_metrics.calculate( error_metr, ego_metrics_true[ego_metr], ego_metric_pred_1)                   
                                        errors_list_1[ego_metr][error_metr].append(error_1)
                                        utils.log_msg('g1 global %s %s = %s' % ( error_metr, ego_metr, error_1 ) )

                                        error_2 = error_metrics.calculate( error_metr, ego_metrics_true[ego_metr], ego_metric_pred_2)                   
                                        errors_list_2[ego_metr][error_metr].append(error_2)
                                        utils.log_msg('g2 global ds %s %s = %s' % ( error_metr, ego_metr, error_2 ) )

                                        error_3 = error_metrics.calculate( error_metr, ego_metrics_true[ego_metr], ego_metric_pred_3)                   
                                        errors_list_3[ego_metr][error_metr].append(error_3)
                                        utils.log_msg('g3 local %s %s = %s' % ( error_metr, ego_metr, error_3 ) )

                        for ego_metr in self.ego_metrics:
                            for error_metr in self.error_met:
                                ego_metric_mean_1 = np.mean( errors_list_1[ego_metr][error_metr])
                                errors_1[ego_metr][error_metr].append(ego_metric_mean_1)

                                ego_metric_mean_2 = np.mean( errors_list_2[ego_metr][error_metr] )
                                errors_2[ego_metr][error_metr].append(ego_metric_mean_2)  

                                ego_metric_mean_3 = np.mean( errors_list_3[ego_metr][error_metr] )
                                errors_3[ego_metr][error_metr].append(ego_metric_mean_3)  

                    legends = ['global', 
                                'global + DS',
                                'local ' ] 
                    
                    for ego_metr in self.ego_metrics:
                        for error_metr in self.error_met: 
                            y = []
                            y.append(errors_1[ego_metr][error_metr])
                            y.append(errors_2[ego_metr][error_metr])
                            y.append(errors_3[ego_metr][error_metr])
                            path_result = "./data/%s/results/result_%s_%s_%s_%s.png" % ( dataset, optin_method, optin_perc, ego_metr, error_metr) 
                            graphics.line_plot2(np.array(self.es), np.array(y), xlabel='$\epsilon$', ylabel= error_metr, ylog=False, line_legends=legends, path=path_result)                                
                            path_result2 = "./data/%s/results/logscale_result_%s_%s_%s_%s.png" % ( dataset, optin_method, optin_perc, ego_metr, error_metr) 
                            graphics.line_plot2(np.array(self.es), np.array(y), xlabel='$\epsilon$', ylabel= error_metr, ylog=True, line_legends=legends, path=path_result2)                                


if __name__ == "__main__":
    datasets_names = [
                    #   'high-school-contacts',
                    #   'copenhagen-interaction',
                    #   'reality-call', 
                      'contacts-dublin' ]
                    #  'digg-reply' ] 
                    # 'wiki-talk',
                    # 'sx-stackoverflow']

    optins_methods = ['affinity']
    optins_perc = [.2]

    es = [ .1, 1, 2 ]

    error_met = ['mre','mae']

    ego_metrics = [ 
                    ## global ##
                    'diameter' ]
                    # 'edges_w',
                    # 'avg_shortest_path' ]

                    # ## ego ##
                    #  'degree',
                    #  'node_strength',
                    #  'node_edges_weight_avg',  
                    #  'sum_of_2_hop_edges',
                    #  'degree_all',
                    #  'node_strength_all', 
                    #  'node_edges_weight_avg_all', 
                    #  'sum_of_2_hop_edges_all',

                    # ## centrality ##
                    #  'pagerank_w',
                    #  'betweenness_w',
                    #  'eigenvector_w',
                    #  'pagerank_w_all',
                    #  'betweenness_w_all',
                    #  'eigenvector_w_all',

                    # ## clustering ##
                    #  'local_clustering_w',
                    #  'global_clustering_w',
                    #  'local_clustering_w_all',
                    #  'global_clustering_w_all'

                    # ]

    runs = 3

    exp = ResultsDPWeightedNets(datasets_names, optins_methods, optins_perc, es, error_met, ego_metrics, runs)
    exp.run()