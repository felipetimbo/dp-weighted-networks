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
                    divide_data_into,
                    runs
                ):
        self.datasets_names = datasets_names  
        self.optins_methods = optins_methods
        self.optins_perc = optins_perc
        self.es = es
        self.error_met = error_met
        self.ego_metrics = ego_metrics
        self.divide_data_into = divide_data_into
        self.runs = runs 

    def run(self):
        
        for dataset in self.datasets_names:
            utils.log_msg('*************** DATASET = ' + dataset + ' ***************')
            for optin_method in self.optins_methods: 
                for optin_perc in self.optins_perc:
                    url = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, '%s_%s_%s.graphml' % (dataset, optin_method, optin_perc )))
                    g = WGraph(url)

                    ego_metrics_true = {}
                    sets_ego_metric_1 = {}
                    sets_ego_metric_2 = {}

                    errors_1 = {} # approach 1
                    errors_2 = {} # approach 2

                    for ego_metric in self.ego_metrics:
                        ego_metrics_true[ego_metric] = egocentric_metrics.calculate(g, ego_metric )
                        # errors_1[ego_metric] = {}
                        # errors_2[ego_metric] = {}

                        # for error_metr in self.error_met: 
                        #     errors_1[ego_metric][error_metr] = {}
                        #     errors_2[ego_metric][error_metr] = {}

                        #     for i in range(self.divide_data_into):
                        #         errors_1[ego_metric][error_metr][i] = []
                        #         errors_2[ego_metric][error_metr][i] = []
                    
                    for e in self.es:
                        utils.log_msg('*************** e = ' + str(e) + ' ***************')
                    
                        errors_list_1 = {} # approach 1
                        errors_list_2 = {} # approach 2

                        for ego_metric in self.ego_metrics:
                            errors_1[ego_metric] = {}
                            errors_2[ego_metric] = {}
                            errors_list_1[ego_metric] = {}
                            errors_list_2[ego_metric] = {}

                            for error_metr in self.error_met: 
                                errors_1[ego_metric][error_metr] = {}
                                errors_2[ego_metric][error_metr] = {}
                                errors_list_1[ego_metric][error_metr] = {}
                                errors_list_2[ego_metric][error_metr] = {}

                                for i in range(self.divide_data_into):
                                    errors_1[ego_metric][error_metr][i] = []
                                    errors_2[ego_metric][error_metr][i] = []
                                    errors_list_1[ego_metric][error_metr][i] = []
                                    errors_list_2[ego_metric][error_metr][i] = []
                                    
                        for r in range(self.runs):    
                            path_g1 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, 'exp', 'graph_perturbed_%s_ins%s_e%s_r%s_baseline.graphml' % ( optin_method, optin_perc, e, r )))
                            g1 = WGraph(path_g1)

                            path_g2 = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, 'exp', 'graph_perturbed_%s_ins%s_e%s_r%s_local.graphml' % ( optin_method, optin_perc, e, r )))
                            g2 = WGraph(path_g2)

                            for ego_metr in self.ego_metrics:
                                ego_metric_pred_1 = egocentric_metrics.calculate(g1, ego_metr)
                                ego_metric_pred_2 = egocentric_metrics.calculate(g2, ego_metr)
                            
                                pos_ego_metric_1 = np.argsort(ego_metrics_true[ego_metr])
                                sets_ego_metric_1[ego_metr] = np.array_split(pos_ego_metric_1, self.divide_data_into)
                                pos_ego_metric_2 = np.argsort(ego_metrics_true[ego_metr])
                                sets_ego_metric_2[ego_metr] = np.array_split(pos_ego_metric_2, self.divide_data_into)

                                for i in range(self.divide_data_into):
                                    ego_metric_pred_1_i = ego_metric_pred_1[sets_ego_metric_1[ego_metr][i]]
                                    ego_metric_pred_2_i = ego_metric_pred_2[sets_ego_metric_2[ego_metr][i]]

                                    for error_metr in self.error_met: 
                                        error_1 = error_metrics.calculate( error_metr, ego_metrics_true[ego_metr][sets_ego_metric_1[ego_metr][i]], ego_metric_pred_1_i)                   
                                        errors_list_1[ego_metr][error_metr][i].append(error_1)
                                        utils.log_msg('g1 global %s %s = %s' % ( error_metr, ego_metr, error_1 ) )

                                        error_2 = error_metrics.calculate( error_metr, ego_metrics_true[ego_metr][sets_ego_metric_2[ego_metr][i]], ego_metric_pred_2_i)                   
                                        errors_list_2[ego_metr][error_metr][i].append(error_2)
                                        utils.log_msg('g2 local %s %s = %s' % ( error_metr, ego_metr, error_2 ) )

                        for ego_metr in self.ego_metrics:
                            for error_metr in self.error_met:
                                for i in range(self.divide_data_into):
                                    ego_metric_mean_1 = np.mean( errors_list_1[ego_metr][error_metr][i] )
                                    errors_1[ego_metr][error_metr][i].append(ego_metric_mean_1)

                                    ego_metric_mean_2 = np.mean( errors_list_2[ego_metr][error_metr][i] )
                                    errors_2[ego_metr][error_metr][i].append(ego_metric_mean_2)  

                        x = list(range(1,divide_data_into + 1))

                        legends = ['global approach', 
                                    'local approach ' ] 
                        
                        for ego_metr in self.ego_metrics:
                            for error_metr in self.error_met: 
                                y = []
                                y.append(errors_1[ego_metr][error_metr].values())
                                y.append(errors_2[ego_metr][error_metr].values())
                                path_result = "./data/%s/results/result_%s_%s_%s_%s_e%s_div%s.png" % ( dataset, optin_method, optin_perc, ego_metr, error_metr, e, self.divide_data_into) 
                                graphics.line_plot2(np.array(x), np.array(y), xlabel='decile', ylabel= error_metr, ylog=False, line_legends=legends, path=path_result)                                
                                # path_result2 = "./data/%s/results/result_%s_%s_%s_%s_e%s_div%s_logscale.png" % ( dataset, optin_method, optin_perc, ego_metr, error_metr, e, self.divide_data_into) 
                                # graphics.line_plot2(np.array(x), np.array(y), xlabel='decile', ylabel= error_metr, ylog=True, line_legends=legends, path=path_result2) 


if __name__ == "__main__":
    datasets_names = [
                    # 'high-school-contacts',
                    # 'copenhagen-interaction',
                    # 'reality-call', 
                    # 'contacts-dublin',
                    'digg-reply' ]
                    # 'wiki-talk',
                    # 'sx-stackoverflow']

    optins_methods = ['affinity']
    optins_perc = [.2]

    es = [ 1 ]

    error_met = ['mre']

    ego_metrics = [ 'sum_of_2_hop_edges',
                     'degree',
                     'node_strength',
                     'node_edges_weight_avg',  
                    #  'density', 
                    #  'num_edges_in_alters' ]
                    # 'density',
                     'density_w',
                    # 'm',
                    # 'total_w',
                    # 'edges_w',
                    #  'degree_all',
                    #  'node_strength_all', 
                    #  'node_edges_weight_avg_all', 
                    # 'num_edges_in_alters_all',
                    #  'sum_of_2_hop_edges_all',
                    # 'ego_betweenness',
                    # 'ego_betweenness_all',
                     'ego_betweenness_w']
                    # 'page_rank',
                    # 'page_rank_all']

    divide_data_into = 10

    runs = 5

    exp = ResultsDPWeightedNets(datasets_names, optins_methods, optins_perc, es, error_met, ego_metrics, divide_data_into, runs)
    exp.run()