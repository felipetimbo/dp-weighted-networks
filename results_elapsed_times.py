import os
import numpy as np
import pandas as pd
import itertools
import pickle

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
                    runs
                ):
        self.datasets_names = datasets_names  
        self.optins_methods = optins_methods
        self.optins_perc = optins_perc
        self.es = es
        self.runs = runs 

    def run(self):
        
        ms = []
        elapsed_times_1 = [] # global

        for dataset in self.datasets_names:
            utils.log_msg('*************** DATASET = ' + dataset + ' ***************')
            for optin_method in self.optins_methods: 
                for optin_perc in self.optins_perc:
                    
                    url = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data', dataset, '%s_%s_%s.graphml' % (dataset, optin_method, optin_perc )))
                    g = WGraph(url)
                    ms.append(g.m())

                    for e in self.es:
                        url = "./data/%s/exp/%s_ins%s_e%s_global.csv" % ( dataset , optin_method, optin_perc, e) 
                        df = pd.read_csv(url) 
                        elapsed_time = df.iloc[0,0] 
                        elapsed_times_1.append(elapsed_time) 

        y = []
        y.append(elapsed_times_1)

        legends = ['global']
        
        path_result = "./data/elapsed_times.png"
        graphics.line_plot2(np.array(ms), np.array(y), xlabel='m', ylabel= "elapsed time (sec)", xlog=True, line_legends=legends, figsize=(5, 5))#, path=path_result)



if __name__ == "__main__":
    datasets_names = [
                         'high-school-contacts',
                         'reality-call2',
                         'enron',
                         'dblp'
                    ]

    optins_methods = ['affinity']
    optins_perc = [.0]

    es = [ 1 ]

    runs = 5

    exp = ResultsDPWeightedNets(datasets_names, optins_methods, optins_perc, es, runs)
    exp.run()