import os
import numpy as np
import pandas as pd

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
                    # errors_2 = {} # approach 2
                    # errors_3 = {} # approach 2
                    # errors_3 = {} # approach 3

                    for ego_metric in self.ego_metrics:
                        ego_metrics_true[ego_metric] = egocentric_metrics.calculate(g, ego_metric )
                        errors_1[ego_metric] = {}
                        # errors_2[ego_metric] = {}
                        # errors_3[ego_metric] = {}
                        # errors_3[ego_metric] = {}

                        for error_metr in self.error_met: 
                            errors_1[ego_metric][error_metr] = []
                            # errors_2[ego_metric][error_metr] = []
                            # errors_3[ego_metric][error_metr] = []
                            # errors_3[ego_metric][error_metr] = []

                    for e in self.es:
                        utils.log_msg('*************** e = ' + str(e) + ' ***************')
                    
                        errors_list_1 = {} # approach 1
                        # errors_list_2 = {} # approach 2
                        # errors_list_3 = {} # approach 2
                        # errors_list_3 = {} # approach 3

                        for ego_metric in self.ego_metrics:
                            errors_list_1[ego_metric] = {}
                            # errors_list_2[ego_metric] = {}
                            # errors_list_3[ego_metric] = {}
                            # errors_list_3[ego_metric] = {}

                            for error_metr in self.error_met: 
                                errors_list_1[ego_metric][error_metr] = []
                                # errors_list_2[ego_metric][error_metr] = []
                                # errors_list_3[ego_metric][error_metr] = []
                                # errors_list_3[ego_metric][error_metr] = []
                                   
                        for r in range(self.runs):    
                            path_g1 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, 'exp', 'graph_perturbed_%s_ins%s_e%s_r%s_global_ds_ns_ins.graphml' % ( optin_method, optin_perc, e, r )))
                            g1 = WGraph(path_g1)

                            # path_g2 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, 'exp', 'graph_perturbed_%s_ins%s_e%s_r%s_global_ds_ns_ins.graphml' % ( optin_method, optin_perc, e, r )))
                            # g2 = WGraph(path_g2)

                            # path_g3 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, 'exp', 'graph_perturbed_%s_ins%s_e%s_r%s_baseline.graphml' % ( optin_method, optin_perc, e, r )))
                            # g3 = WGraph(path_g3)

                            for ego_metr in self.ego_metrics:
                                ego_metric_pred_1 = egocentric_metrics.calculate(g1, ego_metr)
                                # ego_metric_pred_2 = egocentric_metrics.calculate(g2, ego_metr)
                                # ego_metric_pred_3 = egocentric_metrics.calculate(g3, ego_metr)
                            
                                for error_metr in self.error_met: 
                                    if ego_metr == 'edges_w':
                                            
                                        error_1 = error_metrics.calculate_error_edges_w( error_metr, ego_metrics_true[ego_metr], ego_metric_pred_1)                   
                                        errors_list_1[ego_metr][error_metr].append(error_1)
                                        # utils.log_msg('g1 global %s %s = %s' % ( error_metr, ego_metr, error_1 ) )

                                        # error_2 = error_metrics.calculate_error_edges_w( error_metr, ego_metrics_true[ego_metr], ego_metric_pred_2)                   
                                        # errors_list_2[ego_metr][error_metr].append(error_2)
                                        # utils.log_msg('g2 global ds %s %s = %s' % ( error_metr, ego_metr, error_2 ) )

                                        # error_3 = error_metrics.calculate_error_edges_w( error_metr, ego_metrics_true[ego_metr], ego_metric_pred_3)                   
                                        # errors_list_3[ego_metr][error_metr].append(error_3)
                                        # utils.log_msg('g3 local %s %s = %s' % ( error_metr, ego_metr, error_3 ) )
                                    
                                    else:
                                        error_1 = error_metrics.calculate( error_metr, ego_metrics_true[ego_metr], ego_metric_pred_1)                   
                                        errors_list_1[ego_metr][error_metr].append(error_1)
                                        # utils.log_msg('g1 global %s %s = %s' % ( error_metr, ego_metr, error_1 ) )

                                        # error_2 = error_metrics.calculate( error_metr, ego_metrics_true[ego_metr], ego_metric_pred_2)                   
                                        # errors_list_2[ego_metr][error_metr].append(error_2)
                                        # # utils.log_msg('g2 global ds %s %s = %s' % ( error_metr, ego_metr, error_2 ) )

                                        # error_3 = error_metrics.calculate( error_metr, ego_metrics_true[ego_metr], ego_metric_pred_3)                   
                                        # errors_list_3[ego_metr][error_metr].append(error_3)
                                        # utils.log_msg('g3 local %s %s = %s' % ( error_metr, ego_metr, error_3 ) )

                        for ego_metr in self.ego_metrics:
                            for error_metr in self.error_met:
                                ego_metric_mean_1 = np.mean( errors_list_1[ego_metr][error_metr])
                                errors_1[ego_metr][error_metr].append("{:.2f}".format(ego_metric_mean_1)) 

                                # ego_metric_mean_2 = np.mean( errors_list_2[ego_metr][error_metr] )
                                # errors_2[ego_metr][error_metr].append("{:.2f}".format(ego_metric_mean_2))  

                                # ego_metric_mean_3 = np.mean( errors_list_3[ego_metr][error_metr] )
                                # errors_3[ego_metr][error_metr].append("{:.2f}".format(ego_metric_mean_3))  

                    values_concatenated = {}
                    for ego_metr in self.ego_metrics:
                        v1 = errors_1[ego_metr][error_metr]
                        # v2 = errors_2[ego_metr][error_metr]
                        # v3 = errors_3[ego_metr][error_metr]

                        # v12 = np.append(v1, v2, axis=0)
                        # vs = np.append(v12, v3, axis=0)

                        values_concatenated[ego_metr] = v1
                        # values_concatenated[ego_metr] = vs

                    legends = ['global + DS + NS adjustment (PGD)'] 
                                # 'global + DS + greedy NS' , 
                                # # 'global + DS + optimiz.' ] 
                                # 'baseline' ] 

                    # legends = ['global + DS + optimiz. w = 1 ', 
                    #             'global + DS + optimiz. w = 3' , 
                    #             'global + DS + optimiz. w = 3 (fixed)' ] 

                    header = ['metric']
                    for e in self.es*len(legends):
                        header.append('e=' + str(e))

                    header_approach = ['']
                    j = 0
                    for i in range(len(self.es*len(legends))):
                        if i % len(self.es) == 0:
                            header_approach.append(legends[j])
                            j += 1
                        else:
                            header_approach.append('')

                    results = np.empty([ len(self.ego_metrics), len(self.es) * len(legends) + 1 ], dtype=object)
                    for i in range(len(self.ego_metrics)):
                        for j in range(len(self.es) * len(legends) + 1):
                            if j == 0:
                                results[i][j] = self.ego_metrics[i]
                            # elif j == 1:
                            #     results[i][j] = ego_metrics_true[self.ego_metrics[i]]
                            else:
                                results[i][j] = values_concatenated[self.ego_metrics[i]][j-1]
                    
                    path_result = "./data/%s/results/%s_ego_metrics_%s_%s2.csv" % ( dataset, error_metr, optin_method, optin_perc) 
                    df = pd.DataFrame(results) # .to_csv(path_result, header=header, index=False)
                    df.loc[-1] = header
                    df.index = df.index + 1  # shifting index
                    df.sort_index(inplace=True)

                    df.to_csv(path_result, header=header_approach, index=False)

                    print(df)

if __name__ == "__main__":
    datasets_names = [
                          'copenhagen-interaction',
                          'high-school-contacts',
                          'reality-call',
                          'contacts-dublin',
                          'digg-reply', 
                          'enron' 
                      ] 
                    # 'wiki-talk',
                    # 'sx-stackoverflow']

    optins_methods = ['affinity']
    optins_perc = [.2]

    es = [ .5, 1, 2 ]

    error_met = ['mre'] 

    ego_metrics = [ 
                    ## global ##
                    # 'diameter',
                    # 'edges_w',
                    # 'avg_shortest_path' ]

                    ## ego ##
                      'degree',
                      'num_edges_in_alters',

                      'node_strength',
                      'node_edges_weight_avg',  
                    #   'ego_betweenness_w',
                      'sum_of_2_hop_edges',
                    #   'betweenness_w',
                    #   'local_clustering_w',
                    #   'degree_all',
                    #   'num_edges_in_alters_all',
                    #   'node_strength_all', 
                    #   'node_edges_weight_avg_all', 
                    #   'sum_of_2_hop_edges_all']
                

                    # ## centrality ##
                         'pagerank_w',
                         'betweenness_w',
                         'eigenvector_w',

                    #  'pagerank_w_all',
                    #  'betweenness_w_all',
                    #  'eigenvector_w_all',

                    # # ## clustering ##
                        'local_clustering_w',
                    #  'global_clustering_w',
                    #  'local_clustering_w_all',
                    #  'global_clustering_w_all'

                    ]

    runs = 5

    exp = ResultsDPWeightedNets(datasets_names, optins_methods, optins_perc, es, error_met, ego_metrics, runs)
    exp.run()