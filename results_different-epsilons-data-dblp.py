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
                    for ego_metr in self.ego_metrics:

                        x = np.array([0.031, 0.101, 0.097, 0.093, 0.088, 0.079, 0.065, 0.041, 0.128, 0.124, 0.119, 0.113, 0.103, 0.086, 0.058, 0.158, 0.153, 0.145, 0.134, 0.115, 0.079, 0.189, 0.181, 0.167, 0.145, 0.102, 0.215, 0.200, 0.174, 0.124, 0.231, 0.201, 0.143, 0.226, 0.161, 0.175])
                        y = x * 1.8

                        data_4d = np.concatenate((es, np.array([y]).T ), axis=1)

                        path_result = "./data/%s/results/comparison_eps_%s_%s_%s_orig_pert.png" % ( dataset, optin_method, optin_perc, ego_metr) 
                        graphics.scatter_plot4D(data_4d[:,0], data_4d[:,1], data_4d[:,2], data_4d[:,3], xlabel='$\epsilon_1$', ylabel= '$\epsilon_2$', zlabel= '$\epsilon_3$', ylog=False , path=path_result)                                
                        # graphics.surface_plot4D(self.es[:,0], self.es[:,1], self.es[:,2], np.array(y), xlabel='$\epsilon_1$', ylabel= '$\epsilon_2$', zlabel= '$\epsilon_3$', ylog=False, path=path_result)                                


if __name__ == "__main__":
    datasets_names = [
                        #  'high-school-contacts',
                        #  'reality-call2',
                        #  'enron',
                           'dblp'
                    ]

                    # 'wiki-talk'
                    # 'sx-stackoverflow',
                    # 'contacts-dublin',
                    # 'digg-reply',
                    # 'copenhagen-interaction',

                    # 'wiki-talk',
                    # 'sx-stackoverflow']

    optins_methods = ['affinity']
    optins_perc = [.0]

    es = []

    for i in itertools.product ( [1, 2, 3, 4, 5, 6, 7, 8], repeat=3 ):
        if np.sum(i) == 10:
            es.append(i)

    es = np.multiply(es, .1)

    ego_metrics = [ 
                    ## global ##
                    # 'diam',
                    #  'diameter',
                    # 'diameter_w',
                    #  'avg_shortest_path',
                    #   'avg_shortest_path_w',
                    #  'avg_degree',
                    #  'avg_edges_w', 
                    #  'num_triangles',
                    #  'global_clustering_w_all',
                    # 'graph_density',
                    'similarity'
                ]

    runs = 5

    exp = ResultsDPWeightedNets(datasets_names, optins_methods, optins_perc, es, ego_metrics, runs)
    exp.run()