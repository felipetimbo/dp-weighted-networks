import os
from cvxpy import length
import numpy as np
import graph_tool.all as gt

from graph.wgraph import WGraph
from dpwnets import tools
from preproc import statistics
from dpwnets import utils

np.random.seed(0)

class preprocessing():

    def __init__(self, datasets_names, optins_methods, optins_perc):
        self.datasets_names = datasets_names
        self.optins_methods = optins_methods
        self.optins_perc = optins_perc

    def run(self):
        for dataset in self.datasets_names:
            for optin_method in self.optins_methods:
                for optin_perc in self.optins_perc:

                    url = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', dataset, '%s_%s_%s.graphml' % (dataset, optin_method, optin_perc)))
                    g = WGraph(url)

                    utils.log_msg("========== old statistics ==========")
                    g.print_statistics()

                    # utils.log_msg("Adding edges...")
                    # new_g = self.increase_degree(g)

                    # utils.log_msg("Removing edges...")
                    new_g = self.decrease_degree(g)

                    utils.log_msg("========== NEW statistics ==========")
                    new_g.print_statistics()

                    url_output = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', dataset + '2'))

                    # utils.log_msg('Saving %s %s graph (%s %% optins)... ' % (dataset, 'random', str(optin_perc*100)))
                    new_g.save(url_output + '/%s2_%s_%s.graphml' % (dataset, optin_method, str(optin_perc)))

    def decrease_degree(self, g):
        num_edges_to_be_removed = 80000
        edges_g = g.get_edges([g.ep.ew])

        while True: # for i in range(num_edges_to_be_removed):
            probs = (max(edges_g[:,2])+1 - edges_g[:,2])/np.sum(max(edges_g[:,2])+1 - edges_g[:,2])
            new_edges_pos = np.random.choice(len(edges_g), 1, replace=False, p=probs)
            nodes = edges_g[new_edges_pos][0][0:2]
            if g.get_out_degrees([nodes[0]])[0] != 1 and g.get_out_degrees([nodes[1]])[0] != 1:
                edges_g_pos = np.ones( len(edges_g), dtype=bool)
                edges_g_pos[new_edges_pos] = False
                edges_g = edges_g[edges_g_pos]
                g = tools.build_g_from_edges(g, edges_g)
                num_edges_to_be_removed -= 1
                if num_edges_to_be_removed == 0:
                    break
       
        return g

    def increase_degree(self, g):
        prob_geometric = 0.95
        num_edges_to_be_sampled = 80000
        edges_g = g.get_edges([g.ep.ew])
        non_optins_pos = g.vp.optin.fa == 0
        random_edges = tools.sample_random_edges(g.n(), num_edges_to_be_sampled, set(map(tuple, edges_g[:,[0,1]])), non_optins_pos)
        random_weights = np.random.geometric(p=prob_geometric, size=num_edges_to_be_sampled)
        new_edges = np.concatenate((random_edges, np.array([random_weights]).T ), axis=1)
        all_edges = np.concatenate((edges_g, new_edges))
        new_g = tools.build_g_from_edges(g, all_edges)
        return new_g

if __name__ == "__main__":
    datasets_names = [
        # 'reality-call',
        # 'contacts-dublin',
        # 'digg-reply',
         'enron',
        # 'wiki-talk',
        # 'dblp'
    ]

    optins_methods = ['affinity']       # random AND/OR affinity
    optins_perc = [.0]

    preprocessing = preprocessing(datasets_names, optins_methods, optins_perc)
    preprocessing.run()
