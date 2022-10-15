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
                    for ego_metr in self.ego_metrics:
                        for error_metr in self.error_met: 



                            # degrees hsc 
                            # arr_1 = np.array([0.0291, 0.027, 0.0254])
                            # arr_2 = np.array([0.0243, 0.022, 0.0202])
                            # arr_3 = np.array([0.0224, 0.0083, 0.0049])
                            # arr_4 = np.array([0.0225, 0.0085, 0.0052])

                            # degrees reality call 
                            arr_1 = np.array([0.181, 0.174, 0.167])
                            arr_2 = np.array([0.170, 0.162, 0.155])
                            arr_3 = np.array([0.310, 0.083, 0.029])
                            arr_4 = np.array([0.312, 0.085, 0.032])

                            # degrees enron 
                            # arr_1 = np.array([0.54, 0.42, 0.36])
                            # arr_2 = np.array([0.34, 0.22, 0.19])
                            # arr_3 = np.array([0.036, 0.018, 0.009 ])
                            # arr_4 = np.array([0.038, 0.021, 0.013])

                            # degrees dblp 
                            # arr_1 = np.array([0.362, 0.311, 0.288])
                            # arr_2 = np.array([0.144, 0.124, 0.105])
                            # arr_3 = np.array([0.191, 0.068, 0.022 ])
                            # arr_4 = np.array([0.203, 0.071, 0.026])

                            y = []
                            y.append(arr_1)
                            y.append(arr_2)
                            y.append(arr_3)
                            y.append(arr_4)

                            legends = [
                                'high-pass filter ', 
                                'priority sampling',
                                'global approach', 
                                'local approach'
                            ]

                            path_result = "./data/%s/results/result_%s_%s_%s_%s.png" % ( dataset, optin_method, optin_perc, ego_metr, error_metr) 
                            graphics.line_plot3(np.array(self.es), np.array(y), xlabel='$\epsilon$', ylabel= "KL Divergence" if ego_metr != 'similarity' else 'similarity', ylog=False, line_legends=legends, figsize=(5, 5), path=path_result)                                
                            

if __name__ == "__main__":
    datasets_names = [
                            #   'high-school-contacts',
                              'reality-call2',
                            #  'enron',
                            #  'dblp'
                      ] 

    optins_methods = ['affinity']
    optins_perc = [.0]

    es = [ .1, .5, 1 ]

    error_met = ['kld'] 

    ego_metrics = [ 
                       'degree_all',
                    #   'edges_w',  
                    #  'similarity'  
                    ]

    runs = 5

    exp = ResultsDPWeightedNets(datasets_names, optins_methods, optins_perc, es, error_met, ego_metrics, runs)
    exp.run()