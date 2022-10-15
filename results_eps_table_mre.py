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
            utils.log_msg('*************** DATASET = ' +
                          dataset + ' ***************')
            for optin_method in self.optins_methods:
                for optin_perc in self.optins_perc:
                    url = os.path.abspath(os.path.join(os.path.dirname(
                        __file__), 'data', dataset, '%s_%s_%s.graphml' % (dataset, optin_method, optin_perc)))
                    g = WGraph(url)

                    ego_metrics_true = {}

                    errors_1 = {}  # approach 1
                    errors_2 = {}  # approach 2
                    errors_3 = {}  # approach 2
                    errors_4 = {}  # approach 3
                    errors_5 = {}  # approach 5

                    for ego_metric in self.ego_metrics:
                        ego_metrics_true[ego_metric] = egocentric_metrics.calculate(
                            g, ego_metric)
                        errors_1[ego_metric] = {}
                        errors_2[ego_metric] = {}
                        errors_3[ego_metric] = {}
                        errors_4[ego_metric] = {}
                        errors_5[ego_metric] = {}

                        for error_metr in self.error_met:
                            errors_1[ego_metric][error_metr] = []
                            errors_2[ego_metric][error_metr] = []
                            errors_3[ego_metric][error_metr] = []
                            errors_4[ego_metric][error_metr] = []
                            errors_5[ego_metric][error_metr] = []

                    for e in self.es:
                        utils.log_msg('*************** e = ' +
                                      str(e) + ' ***************')

                        errors_list_1 = {}  # approach 1
                        errors_list_2 = {}  # approach 2
                        errors_list_3 = {}  # approach 2
                        errors_list_4 = {}  # approach 3
                        errors_list_5 = {}  # approach 3

                        for ego_metric in self.ego_metrics:
                            errors_list_1[ego_metric] = {}
                            errors_list_2[ego_metric] = {}
                            errors_list_3[ego_metric] = {}
                            errors_list_4[ego_metric] = {}
                            errors_list_5[ego_metric] = {}

                            for error_metr in self.error_met:
                                errors_list_1[ego_metric][error_metr] = []
                                errors_list_2[ego_metric][error_metr] = []
                                errors_list_3[ego_metric][error_metr] = []
                                errors_list_4[ego_metric][error_metr] = []
                                errors_list_5[ego_metric][error_metr] = []

                        for r in range(self.runs):
                            path_g1 = os.path.abspath(os.path.join(os.path.dirname(
                                __file__), 'data', dataset, 'exp', '%s_ins%s_e%s_r%s_baseline2.graphml' % (optin_method, optin_perc, e, r)))
                            g1 = WGraph(path_g1)

                            path_g2 = os.path.abspath(os.path.join(os.path.dirname(
                                __file__), 'data', dataset, 'exp', '%s_ins%s_e%s_t%s_r%s_baseline3.graphml' % (optin_method, optin_perc, e, 7, r)))
                            g2 = WGraph(path_g2)

                            path_g3 = os.path.abspath(os.path.join(os.path.dirname(
                                __file__), 'data', dataset, 'exp', '%s_ins%s_e%s_r%s_baseline4.graphml' % (optin_method, optin_perc, e, r)))
                            g3 = WGraph(path_g3)

                            path_g4 = os.path.abspath(os.path.join(os.path.dirname(
                                __file__), 'data', dataset, 'exp', '%s_ins%s_e%s_r%s_global.graphml' % (optin_method, optin_perc, e, r)))
                            g4 = WGraph(path_g4)

                            path_g5 = os.path.abspath(os.path.join(os.path.dirname(
                                __file__), 'data', dataset, 'exp', '%s_ins%s_e%s_r%s_local.graphml' % (optin_method, optin_perc, e, r)))
                            g5 = WGraph(path_g5)

                            for ego_metr in self.ego_metrics:
                                ego_metric_pred_1 = egocentric_metrics.calculate(
                                    g1, ego_metr)
                                ego_metric_pred_2 = egocentric_metrics.calculate(
                                    g2, ego_metr)
                                ego_metric_pred_3 = egocentric_metrics.calculate(
                                    g3, ego_metr)
                                ego_metric_pred_4 = egocentric_metrics.calculate(
                                    g4, ego_metr)
                                ego_metric_pred_5 = egocentric_metrics.calculate(
                                    g5, ego_metr)

                                for error_metr in self.error_met:
                                    if ego_metr == 'edges_w':

                                        error_1 = error_metrics.calculate_error_edges_w(
                                            error_metr, ego_metrics_true[ego_metr], ego_metric_pred_1)
                                        errors_list_1[ego_metr][error_metr].append(
                                            error_1)
                                        # utils.log_msg('g1 global %s %s = %s' % ( error_metr, ego_metr, error_1 ) )

                                        error_2 = error_metrics.calculate_error_edges_w(
                                            error_metr, ego_metrics_true[ego_metr], ego_metric_pred_2)
                                        errors_list_2[ego_metr][error_metr].append(
                                            error_2)
                                        # utils.log_msg('g2 global ds %s %s = %s' % ( error_metr, ego_metr, error_2 ) )

                                        error_3 = error_metrics.calculate_error_edges_w(
                                            error_metr, ego_metrics_true[ego_metr], ego_metric_pred_3)
                                        errors_list_3[ego_metr][error_metr].append(
                                            error_3)
                                        # utils.log_msg('g3 local %s %s = %s' % ( error_metr, ego_metr, error_3 ) )

                                        error_4 = error_metrics.calculate_error_edges_w(
                                            error_metr, ego_metrics_true[ego_metr], ego_metric_pred_4)
                                        errors_list_4[ego_metr][error_metr].append(
                                            error_4)
                                        # utils.log_msg('g3 local %s %s = %s' % ( error_metr,

                                        error_5 = error_metrics.calculate_error_edges_w(
                                            error_metr, ego_metrics_true[ego_metr], ego_metric_pred_5)
                                        errors_list_5[ego_metr][error_metr].append(
                                            error_5)

                                    else:
                                        error_1 = error_metrics.calculate(
                                            error_metr, ego_metrics_true[ego_metr], ego_metric_pred_1)
                                        errors_list_1[ego_metr][error_metr].append(
                                            error_1)
                                        # utils.log_msg('g1 global %s %s = %s' % ( error_metr, ego_metr, error_1 ) )

                                        error_2 = error_metrics.calculate(
                                            error_metr, ego_metrics_true[ego_metr], ego_metric_pred_2)
                                        errors_list_2[ego_metr][error_metr].append(
                                            error_2)
                                        # # utils.log_msg('g2 global ds %s %s = %s' % ( error_metr, ego_metr, error_2 ) )

                                        error_3 = error_metrics.calculate(
                                            error_metr, ego_metrics_true[ego_metr], ego_metric_pred_3)
                                        errors_list_3[ego_metr][error_metr].append(
                                            error_3)
                                        # utils.log_msg('g3 local %s %s = %s' % ( error_metr, ego_metr, error_3 ) )

                                        error_4 = error_metrics.calculate(
                                            error_metr, ego_metrics_true[ego_metr], ego_metric_pred_4)
                                        errors_list_4[ego_metr][error_metr].append(
                                            error_4)

                                        error_5 = error_metrics.calculate(
                                            error_metr, ego_metrics_true[ego_metr], ego_metric_pred_5)
                                        errors_list_5[ego_metr][error_metr].append(
                                            error_5)

                        for ego_metr in self.ego_metrics:
                            for error_metr in self.error_met:
                                ego_metric_mean_1 = np.mean(
                                    errors_list_1[ego_metr][error_metr])
                                errors_1[ego_metr][error_metr].append(
                                    float("{:.2f}".format(ego_metric_mean_1)))

                                ego_metric_mean_2 = np.mean(
                                    errors_list_2[ego_metr][error_metr])
                                errors_2[ego_metr][error_metr].append(
                                    float("{:.2f}".format(ego_metric_mean_2)))

                                ego_metric_mean_3 = np.mean(
                                    errors_list_3[ego_metr][error_metr])
                                errors_3[ego_metr][error_metr].append(
                                    float("{:.2f}".format(ego_metric_mean_3)))

                                ego_metric_mean_4 = np.mean(
                                    errors_list_4[ego_metr][error_metr])
                                errors_4[ego_metr][error_metr].append(
                                    float("{:.2f}".format(ego_metric_mean_4)))

                                ego_metric_mean_5 = np.mean(
                                    errors_list_5[ego_metr][error_metr])
                                errors_5[ego_metr][error_metr].append(
                                    float("{:.2f}".format(ego_metric_mean_5)))

                    values_concatenated = {}
                    for ego_metr in self.ego_metrics:
                        v1 = errors_1[ego_metr][error_metr]
                        v2 = errors_2[ego_metr][error_metr]
                        v3 = errors_3[ego_metr][error_metr]
                        v4 = errors_4[ego_metr][error_metr]
                        v5 = errors_5[ego_metr][error_metr]

                        v12 = np.append(v1, v2, axis=0)
                        v123 = np.append(v12, v3, axis=0)
                        v1234 = np.append(v123, v4, axis=0)
                        vs = np.append(v1234, v5, axis=0)

                        values_concatenated[ego_metr] = vs

                    legends = [
                        'log-laplace',
                        'truncation $\u03B8 = \Delta W^{1/3}$',
                        'priority sampling',
                        'global approach',
                        'local approach'
                    ]

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

                    results = np.empty([len(self.ego_metrics), len(
                        self.es) * len(legends) + 1], dtype=object)
                    for i in range(len(self.ego_metrics)):
                        for j in range(len(self.es) * len(legends) + 1):
                            if j == 0:
                                results[i][j] = self.ego_metrics[i]
                            # elif j == 1:
                            #     results[i][j] = ego_metrics_true[self.ego_metrics[i]]
                            else:
                                results[i][j] = values_concatenated[self.ego_metrics[i]][j-1]

                    path_result = "./data/%s/results/statistics3_%s_%s.csv" % (
                        dataset, optin_method, optin_perc)
                    # .to_csv(path_result, header=header, index=False)
                    df = pd.DataFrame(results)
                    df.loc[-1] = header
                    df.index = df.index + 1  # shifting index
                    df.sort_index(inplace=True)

                    df.to_csv(path_result, header=header_approach, index=False)

                    print(df)

                    # for ego_metr in self.ego_metrics:
                    #     for error_metr in self.error_met:

                    #         y = []
                    #         y.append(errors_1[ego_metr][error_metr])
                    #         y.append(errors_2[ego_metr][error_metr])
                    #         y.append(errors_3[ego_metr][error_metr])
                    #         y.append(errors_4[ego_metr][error_metr])
                    # path_result = "./data/%s/results/result_%s_%s_%s_%s.png" % ( dataset, optin_method, optin_perc, ego_metr, error_metr)
                    # graphics.line_plot2(np.array(self.es), np.array(y), xlabel='$\epsilon$', ylabel= "MRE", ylog=False, line_legends=legends, figsize=(5, 5), path=path_result)
                    # path_result2 = "./data/%s/results/result_%s_%s_%s_%s_logscale.png" % ( dataset, optin_method, optin_perc, ego_metr, error_metr)
                    # graphics.line_plot2(np.array(self.es), np.array(y), xlabel='$\epsilon$', ylabel= error_metr, ylog=True, line_legends=legends, figsize=(5, 5), path=path_result2)

                    # path_result = "./data/paper/result_%s_%s_%s_%s_%s.png" % ( dataset, optin_method, optin_perc, ego_metr, error_metr)
                    # graphics.line_plot2(np.array(self.es), np.array(y), xlabel='$\epsilon$', ylabel= "KL Divergence", ylog=False, line_legends=legends, figsize=(5, 5), path=path_result)


if __name__ == "__main__":
    datasets_names = [
        # 'high-school-contacts',
        #    'reality-call2',
        #   'enron',
          'dblp'
    ]

    optins_methods = ['affinity']
    optins_perc = [.0]

    es = [.1, .5, 1]

    error_met = ['mre']

    ego_metrics = [
        ## global ##
        # 'diameter',
        # 'edges_w',
        # 'avg_shortest_path' ]

        ## ego ##
        #   'degree',
        #   'num_edges_in_alters',

        #   'node_strength',
        #   'node_edges_weight_avg',
        #    'ego_betweenness_w',
        #   'sum_of_2_hop_edges',
        #   'betweenness_w',
        #   'local_clustering_w',
        #   'degree_all',
        #    'num_edges_in_alters_all',
        'node_strength_all',
        #   'node_edges_weight_avg_all',
        'sum_of_2_hop_edges_all',


        # ## centrality ##
        # 'pagerank_w',
        #  'betweenness_w',
        #  'eigenvector_w',

        'pagerank_w_all',
        #    'betweenness_w_all',
        # 'eigenvector_w_all',

        # # ## clustering ##
        #    'local_clustering_w',
        #  'global_clustering_w',
        # 'local_clustering_w_all',
        #   'global_clustering_w_all'
    ]

    runs = 1

    exp = ResultsDPWeightedNets(
        datasets_names, optins_methods, optins_perc, es, error_met, ego_metrics, runs)
    exp.run()
