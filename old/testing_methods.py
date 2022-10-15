import numpy as np
import graph_tool.all as gt

from metrics import egocentric_metrics
from graph import wgraph

if __name__ == "__main__":
    synthGraph = gt.Graph(directed=False) 
    v = [synthGraph.add_vertex() for i in range(9)]

    optin = synthGraph.new_vertex_property('bool')
    # optin[0] = False
    # optin[1] = True
    # optin[2] = False
    # optin[3] = True
    # optin[4] = True
    # optin[5] = True
    # optin[6] = False
    # optin[7] = False
    # optin[8] = True   
    optin[0] = False
    optin[1] = False
    optin[2] = False
    optin[3] = False
    optin[4] = False
    optin[5] = False
    optin[6] = False
    optin[7] = False
    optin[8] = False   
    synthGraph.vertex_properties['optin'] = optin

    e0=synthGraph.add_edge(v[0], v[1])
    e1=synthGraph.add_edge(v[0], v[3])
    e2=synthGraph.add_edge(v[0], v[4])
    e3=synthGraph.add_edge(v[0], v[7])
    e4=synthGraph.add_edge(v[1], v[2])
    e5=synthGraph.add_edge(v[1], v[4])
    e6=synthGraph.add_edge(v[2], v[5])
    e7=synthGraph.add_edge(v[2], v[6])
    e8=synthGraph.add_edge(v[3], v[4])
    e9=synthGraph.add_edge(v[3], v[6])
    e10=synthGraph.add_edge(v[4], v[5])
    e11=synthGraph.add_edge(v[5], v[7])
    e12=synthGraph.add_edge(v[5], v[8])
    e13=synthGraph.add_edge(v[6], v[7])
    e14=synthGraph.add_edge(v[7], v[8])   

    ew = synthGraph.new_edge_property('int')
    ew[e0]=1
    ew[e1]=4
    ew[e2]=2
    ew[e3]=4
    ew[e4]=1
    ew[e5]=2
    ew[e6]=3
    ew[e7]=7
    ew[e8]=2
    ew[e9]=5
    ew[e10]=3
    ew[e11]=2
    ew[e12]=1
    ew[e13]=3
    ew[e14]=2
    synthGraph.edge_properties['ew'] = ew

    g = wgraph.WGraph(G=synthGraph)

    # g = gt.Graph(G, directed=False)
    egocentric_metrics.strong_shortest_paths_random([g], 5, 3) 