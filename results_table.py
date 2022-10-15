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

                            # edges_w
                            # arr_1 = np.array([1.23, 0.56, 0.29])
                            # arr_2 = np.array([0.52, 0.34, 0.26])
                            # arr_3 = np.array([0.37, 0.29, 0.27])
                            # arr_4 = np.array([0.34, 0.25, 0.22])
                            # arr_5 = np.array([0.35, 0.30, 0.27])
                            # arr_6 = np.array([0.38, 0.11, 0.06])
                            # arr_7 = np.array([0.06, 0.04, 0.02])
                            # arr_8 = np.array([0.09, 0.06, 0.03])

                            # edges_w dblp 
                            # arr_1 = np.array([1.73596, 1.28358, 1.08882])
                            # arr_2 = np.array([1.15993, 0.94014, 0.83582])
                            # arr_3 = np.array([0.83329, 0.50384, 0.3465 ])
                            # arr_4 = np.array([0.66346, 0.36124, 0.27661])
                            # arr_5 = np.array([0.93866, 0.59995, 0.42951])
                            # arr_6 = np.array([0.28968, 0.26144, 0.24415])
                            # arr_7 = np.array([0.08633, 0.05518, 0.03974]) 
                            # arr_8 = np.array([0.08985, 0.06396, 0.04433])

                            # SIMILARITY dblp 
                            arr_1 = np.array([0.019, 0.065, 0.092])
                            arr_2 = np.array([0.026, 0.071, 0.094])
                            arr_3 = np.array([0.061, 0.135, 0.256])
                            arr_4 = np.array([0.063, 0.254, 0.362])
                            arr_5 = np.array([0.052, 0.143, 0.282])
                            arr_6 = np.array([0.028, 0.164, 0.295])
                            arr_7 = np.array([0.073, 0.284, 0.412])
                            arr_8 = np.array([0.069, 0.266, 0.390])
                           
                            y = []
                            y.append(arr_1)
                            y.append(arr_2)
                            y.append(arr_3)
                            y.append(arr_4)
                            y.append(arr_5)
                            y.append(arr_6)
                            y.append(arr_7)
                            y.append(arr_8)

                            legends = [
                                'geometric ', 
                                'log-laplace',
                                'truncation $\u03B8 = \Delta W^{1/4}$',
                                'truncation $\u03B8 = \Delta W^{1/3}$',
                                'truncation $\u03B8 = \Delta W^{1/2}$',
                                'priority sampling',
                                'global approach', 
                                'local approach'
                            ]

                            path_result = "./data/%s/results/result_%s_%s_%s_%s.png" % ( dataset, optin_method, optin_perc, ego_metr, error_metr) 
                            graphics.line_plot2(np.array(self.es), np.array(y), xlabel='$\epsilon$', ylabel= "KL Divergence" if ego_metr != 'similarity' else 'similarity', ylog=False, line_legends=legends, figsize=(5, 5), path=path_result)                                
                            

if __name__ == "__main__":
    datasets_names = [
                            #   'high-school-contacts',
                            #  'reality-call2',
                            #  'enron',
                            'dblp'
                      ] 

    optins_methods = ['affinity']
    optins_perc = [.0]

    es = [ .1, .5, 1 ]

    error_met = ['kld'] 

    ego_metrics = [ 
                    #   'degree_all',
                    #   'edges_w',  
                      'similarity'  
                    ]

    runs = 5

    exp = ResultsDPWeightedNets(datasets_names, optins_methods, optins_perc, es, error_met, ego_metrics, runs)
    exp.run()