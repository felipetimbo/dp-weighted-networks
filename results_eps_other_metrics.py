import os
import numpy as np
import pandas as pd

from dpwnets import (utils, graphics)
from metrics import (error_metrics, egocentric_metrics)
from itertools import chain

from graph.wgraph import WGraph

class ResultsDPWeightedNets():

    def __init__(self, 
                    datasets_names, 
                    optins_methods, 
                    optins_perc, 
                    es, 
                    ego_metrics,
                    runs
                ):
        self.datasets_names = datasets_names  
        self.optins_methods = optins_methods
        self.optins_perc = optins_perc
        self.es = es
        self.ego_metrics = ego_metrics
        self.runs = runs 

    def run(self):
        
        for dataset in self.datasets_names:
            utils.log_msg('*************** DATASET = ' + dataset + ' ***************')
            for optin_method in self.optins_methods: 
                for optin_perc in self.optins_perc:
                    url = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, '%s_%s_%s.graphml' % (dataset, optin_method, optin_perc )))
                    g = WGraph(url, compute_distances=False)

                    ego_metrics_true = {}

                    values_1 = {} # approach 1
                    values_2 = {} # approach 2
                    values_3 = {} # approach 3
                    values_4 = {} # approach 4
                    values_5 = {} # approach 4

                    for ego_metric in self.ego_metrics:
                        if ego_metric != 'similarity':
                            ego_metrics_true[ego_metric] = "{:.5f}".format(egocentric_metrics.calculate(g, ego_metric)) 
                        else:
                            ego_metrics_true[ego_metric] = 1.00

                        values_1[ego_metric] = []
                        values_2[ego_metric] = []
                        values_3[ego_metric] = []
                        values_4[ego_metric] = []
                        values_5[ego_metric] = []

                        # for error_metr in self.error_met: 
                        #     errors_1[ego_metric][error_metr] = []
                        #     errors_2[ego_metric][error_metr] = []
                            # errors_3[ego_metric][error_metr] = []

                    for e in self.es:
                        utils.log_msg('*************** e = ' + str(e) + ' ***************')
                    
                        values_list_1 = {} # approach 1
                        values_list_2 = {} # approach 2
                        values_list_3 = {} # approach 3
                        values_list_4 = {} # approach 4
                        values_list_5 = {} # approach 4

                        for ego_metric in self.ego_metrics:
                            values_list_1[ego_metric] = []
                            values_list_2[ego_metric] = []
                            values_list_3[ego_metric] = []
                            values_list_4[ego_metric] = []
                            values_list_5[ego_metric] = []

                            # for error_metr in self.error_met: 
                            #     errors_list_1[ego_metric][error_metr] = []
                            #     errors_list_2[ego_metric][error_metr] = []
                                # errors_list_3[ego_metric][error_metr] = []
                                   
                        for r in range(self.runs):    
                            path_g1 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, 'exp', '%s_ins%s_e%s_r%s_baseline2.graphml' % ( optin_method, optin_perc, e, r )))
                            g1 = WGraph(path_g1)

                            path_g2 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, 'exp', '%s_ins%s_e%s_t%s_r%s_baseline3.graphml' % ( optin_method, optin_perc, e, 7, r )))
                            g2 = WGraph(path_g2)

                            path_g3 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, 'exp', '%s_ins%s_e%s_r%s_baseline4.graphml' % ( optin_method, optin_perc, e, r )))
                            g3 = WGraph(path_g3)

                            path_g4 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, 'exp', '%s_ins%s_e%s_r%s_global.graphml' % ( optin_method, optin_perc, e, r )))
                            g4 = WGraph(path_g4)

                            path_g5 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, 'exp', '%s_ins%s_e%s_r%s_local.graphml' % ( optin_method, optin_perc, e, r )))
                            g5 = WGraph(path_g5)

                            for ego_metr in self.ego_metrics:
                                if ego_metr == 'similarity':
                                    value_1 = egocentric_metrics.similar(g, g1) 
                                    values_list_1[ego_metr].append(value_1)

                                    value_2 = egocentric_metrics.similar(g, g2) 
                                    values_list_2[ego_metr].append(value_2)

                                    value_3 = egocentric_metrics.similar(g, g3) 
                                    values_list_3[ego_metr].append(value_3)

                                    value_4 = egocentric_metrics.similar(g, g4) 
                                    values_list_4[ego_metr].append(value_4)

                                    value_5 = egocentric_metrics.similar(g, g5) 
                                    values_list_5[ego_metr].append(value_5)

                                # ego_metric_pred_1 = egocentric_metrics.calculate(g1, ego_metr)
                                # ego_metric_pred_2 = egocentric_metrics.calculate(g2, ego_metr)
                                # ego_metric_pred_3 = egocentric_metrics.calculate(g3, ego_metr)                           

                                else: 
                                    value_1 = egocentric_metrics.calculate(g1, ego_metr )                   
                                    values_list_1[ego_metr].append(value_1)

                                    value_2 = egocentric_metrics.calculate(g2, ego_metr )                   
                                    values_list_2[ego_metr].append(value_2)

                                    value_3 = egocentric_metrics.calculate(g3, ego_metr )                   
                                    values_list_3[ego_metr].append(value_3)

                                    value_4 = egocentric_metrics.calculate(g4, ego_metr )                   
                                    values_list_4[ego_metr].append(value_4)

                                    value_5 = egocentric_metrics.calculate(g5, ego_metr )                   
                                    values_list_5[ego_metr].append(value_5)

                                # utils.log_msg('g1 global %s %s = %s' % ( error_metr, ego_metr, error_1 ) )

                                # value_2 = egocentric_metrics.calculate(g2, ego_metr )                    
                                # values_list_2[ego_metr].append(value_2)
                                # utils.log_msg('g2 global ds %s %s = %s' % ( error_metr, ego_metr, error_2 ) )

                                # error_3 = error_metrics.calculate( error_metr, ego_metrics_true[ego_metr], ego_metric_pred_3)                   
                                # errors_list_3[ego_metr][error_metr].append(error_3)
                                # utils.log_msg('g3 local %s %s = %s' % ( error_metr, ego_metr, error_3 ) )

                        for ego_metr in self.ego_metrics:
                            if ego_metr == "graph_density":
                                ego_metric_mean_1 = np.mean( values_list_1[ego_metr])
                                values_1[ego_metr].append("{:.8f}".format(ego_metric_mean_1) )

                                ego_metric_mean_2 = np.mean( values_list_2[ego_metr] )
                                values_2[ego_metr].append("{:.8f}".format(ego_metric_mean_2))  

                                ego_metric_mean_3 = np.mean( values_list_3[ego_metr] )
                                values_3[ego_metr].append("{:.8f}".format(ego_metric_mean_3))  

                                ego_metric_mean_4 = np.mean( values_list_4[ego_metr] )
                                values_4[ego_metr].append("{:.8f}".format(ego_metric_mean_4))  

                                ego_metric_mean_5 = np.mean( values_list_5[ego_metr] )
                                values_5[ego_metr].append("{:.8f}".format(ego_metric_mean_5))  
                            else:
                                ego_metric_mean_1 = np.mean( values_list_1[ego_metr])
                                values_1[ego_metr].append("{:.2f}".format(ego_metric_mean_1) )

                                ego_metric_mean_2 = np.mean( values_list_2[ego_metr] )
                                values_2[ego_metr].append("{:.2f}".format(ego_metric_mean_2))  

                                ego_metric_mean_3 = np.mean( values_list_3[ego_metr] )
                                values_3[ego_metr].append("{:.2f}".format(ego_metric_mean_3))  

                                ego_metric_mean_4 = np.mean( values_list_4[ego_metr] )
                                values_4[ego_metr].append("{:.2f}".format(ego_metric_mean_4))  

                                ego_metric_mean_5 = np.mean( values_list_5[ego_metr] )
                                values_5[ego_metr].append("{:.2f}".format(ego_metric_mean_5)) 


                                # ego_metric_mean_3 = np.mean( errors_list_3[ego_metr][error_metr] )
                                # errors_3[ego_metr][error_metr].append(ego_metric_mean_3)  

                        # es_dict1[e] = values_1
                        # es_dict2[e] = values_2

                    values_concatenated = {}
                    for ego_metr in self.ego_metrics:
                        v1 = values_1[ego_metr]
                        v2 = values_2[ego_metr]
                        v3 = values_3[ego_metr]
                        v4 = values_4[ego_metr]
                        v5 = values_5[ego_metr]

                        v12 = np.append(v1, v2, axis=0)
                        v123 = np.append(v12, v3, axis=0)
                        v1234 = np.append(v123, v4, axis=0)
                        vs = np.append(v1234, v5, axis=0)

                        # values_concatenated[ego_metr] = v12
                        values_concatenated[ego_metr] = vs

                    legends = [
                                'log-laplace',
                                'truncation $\u03B8 = \Delta W^{1/3}$',
                                'priority sampling',
                                'global approach', 
                                'local approach'
                                ] 

                    header = ['metric', 'original']
                    for e in self.es*len(legends):
                        header.append('e=' + str(e))

                    header_approach = ['','']
                    j = 0
                    for i in range(len(self.es*len(legends))):
                        if i % len(self.es) == 0:
                            header_approach.append(legends[j])
                            j += 1
                        else:
                            header_approach.append('')

                    results = np.empty([ len(self.ego_metrics), len(self.es) * len(legends) + 2 ], dtype=object)
                    for i in range(len(self.ego_metrics)):
                        for j in range(len(self.es) * len(legends) + 2):
                            if j == 0:
                                results[i][j] = self.ego_metrics[i]
                            elif j == 1:
                                results[i][j] = ego_metrics_true[self.ego_metrics[i]]
                            else:
                                results[i][j] = values_concatenated[self.ego_metrics[i]][j-2]
                    
                    path_result = "./data/%s/results/statistics2_%s_%s.csv" % ( dataset, optin_method, optin_perc) 
                    # path_result = "./data/paper/statistics2_%s_%s_%s.csv" % ( dataset, optin_method, optin_perc) 

                    df = pd.DataFrame(results) # .to_csv(path_result, header=header, index=False)
                    df.loc[-1] = header
                    df.index = df.index + 1  # shifting index
                    df.sort_index(inplace=True)

                    df.to_csv(path_result, header=header_approach, index=False)

                    print(df)

                    # for ego_metr in self.ego_metrics:

                    #     y = []
                    #     y.append(values_1[ego_metr])
                    #     y.append(values_2[ego_metr])
                    #     y.append(values_3[ego_metr])
                    #     y.append(values_4[ego_metr])

                    #     path_result = "./data/%s/results/result_%s_%s_%s_orig_pert.png" % ( dataset, optin_method, optin_perc, ego_metr) 
                    #     graphics.line_plot2(np.array(self.es), np.array(y), xlabel='$\epsilon$', ylabel= ego_metr, ylog=False, line_legends=legends, figsize=(5, 5), path=path_result)                                


if __name__ == "__main__":
    datasets_names = [
                            #  'high-school-contacts',
                            #   'reality-call2',
                            #   'enron',
                             'dblp'
                      ] 


    optins_methods = ['affinity']
    optins_perc = [.0]

    es = [ .1, .5, 1 ]

    ego_metrics = [ 
                    ## global ##
                    # 'diam',
                    # 'diameter',
                    # 'diameter_w',
                    #    'avg_shortest_path',
                       'total_w',
                       'avg_shortest_path_w',
                    #  'avg_degree',
                    #  'avg_edges_w', 
                    #  'num_triangles',
                      'global_clustering_w_all'
                    # 'graph_density',
                    # 'similarity'
                ]

    runs = 1

    exp = ResultsDPWeightedNets(datasets_names, optins_methods, optins_perc, es, ego_metrics, runs)
    exp.run()