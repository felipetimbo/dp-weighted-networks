import os
import numpy as np
import graph_tool.all as gt

from utils import messages as msgs
from numpy import genfromtxt

np.random.seed(0)

class preproc_data():
    
    def __init__(self, datasets_names, optins_method, optins_perc):
        self.datasets_names = datasets_names
        self.optins_method = optins_method
        self.optins_perc = optins_perc

    def run(self):
        for dataset in self.datasets_names:
            url = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', dataset ))
            edges = genfromtxt('%s/%s.csv' % (url, dataset), delimiter=',' )
            
            g = gt.Graph(directed=False) 
            ew = g.new_edge_property('int') # edge weight
            g.edge_properties['ew'] = ew 

            msgs.log("Adding edges...")
            g.add_edge_list(edges, eprops=[g.ep.ew])

            for optin_method in self.optins_method: 
                for optin_perc in self.optins_perc:
                    if optin_method == 'random':
                        self.optins_based_on_random_selection(g, optin_perc, dataset, url)
                    elif optin_method == 'affinity':
                        self.optins_based_on_affinity_model(g, optin_perc, dataset, url)
                    
    def optins_based_on_random_selection(self, g, optin_perc, dataset, url):
        optin = g.new_vertex_property('bool') 
        g.vertex_properties['optin'] = optin 
        num_optins = int(g.num_vertices()*optin_perc) 
        optins_idx = np.random.choice(g.num_vertices(), num_optins, replace=False) 
        optins_mask = np.zeros(g.num_vertices(), dtype=bool)
        optins_mask[optins_idx] = True
        g.vp.optin.fa = optins_mask

        msgs.log('Saving %s %s graph (%s %% optins)... ' % (dataset, 'random', str(optin_perc*100)))
        g.save(url + '/%s_random_%s.graphml' % (dataset, str(optin_perc)))

    def optins_based_on_affinity_model(self, g, optin_perc, dataset, url):
        perc_seeds = .01

        optin = g.new_vertex_property('bool')
        g.vertex_properties['optin'] = optin

        degrees = g.get_out_degrees(g.get_vertices())
        percentile = np.percentile(degrees, 100 - perc_seeds*100) # 1% seeds
        perc_idx = np.where(degrees >= percentile)[0]
        exceding_values = len(perc_idx) - int(g.num_vertices() * perc_seeds)
        exceding_idx = np.random.choice(perc_idx, exceding_values, replace=False)
        perc_mask_ids = np.array(list(set(perc_idx).difference(set(exceding_idx))))
        optins_mask = np.zeros(g.num_vertices(), dtype=bool)
        optins_mask[perc_mask_ids] = True
        g.vp.optin.fa = optins_mask

        reamining_optins = int(g.num_vertices()*optin_perc) - len(perc_mask_ids)
        while reamining_optins > 0: 
            picked_optin = np.random.choice(perc_mask_ids, 1, replace=True)[0]
            neighbors = g.get_all_neighbors(picked_optin, vprops=[g.vp.optin])
            neighbors_optout = neighbors[neighbors[:,1] == 0][:,0]
            if len(neighbors_optout) > 0: 
                picked_to_be_a_new_optin = np.random.choice(neighbors_optout, 1, replace=False)[0]
                g.vp.optin.fa[picked_to_be_a_new_optin] = True 
                perc_mask_ids = np.append(perc_mask_ids, np.array(picked_to_be_a_new_optin)) 
                reamining_optins -= 1  

        msgs.log('Saving %s %s graph (%s %% optins)... ' % (dataset, 'affinity', str(optin_perc*100)))
        g.save(url + '/%s_affinity_%s.graphml' % (dataset, str(optin_perc)))



if __name__ == "__main__":
    datasets_names = ['copenhagen-interaction',
                    'reality-call']
    optins_methods = ['random', 'affinity']
    optins_perc = [.2, .4, .6, .8]

    preprocessing = preproc_data(datasets_names, optins_methods, optins_perc)
    preprocessing.run()
    