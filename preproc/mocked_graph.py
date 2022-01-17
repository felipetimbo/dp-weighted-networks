import graph_tool.all as gt
import numpy as np

from graph.wgraph import WGraph

class MockedGraph(object):
    
    def __init__(self):

        G = gt.Graph(directed=False) 
        v = [G.add_vertex() for i in range(9)]

        optin = G.new_vertex_property('bool')
        optin[0] = False
        optin[1] = True
        optin[2] = False
        optin[3] = True
        optin[4] = True
        optin[5] = True
        optin[6] = False
        optin[7] = False
        optin[8] = True   

        e0=G.add_edge(v[0], v[1])
        e1=G.add_edge(v[0], v[3])
        e2=G.add_edge(v[0], v[4])
        e3=G.add_edge(v[0], v[7])
        e4=G.add_edge(v[1], v[2])
        e5=G.add_edge(v[1], v[4])
        e6=G.add_edge(v[2], v[5])
        e7=G.add_edge(v[2], v[6])
        e8=G.add_edge(v[3], v[4])
        e9=G.add_edge(v[3], v[6])
        e10=G.add_edge(v[4], v[5])
        e11=G.add_edge(v[5], v[7])
        e12=G.add_edge(v[5], v[8])
        e13=G.add_edge(v[6], v[7])
        e14=G.add_edge(v[7], v[8])   

        ew = G.new_edge_property('int')
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

        G.edge_properties['ew'] = ew
        G.vertex_properties['optin'] = optin
        self.G = G

    def __getattr__(self, *args, **kwargs):
        return getattr(self.G, *args, **kwargs)