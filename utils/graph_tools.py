from collections import defaultdict

class Graph():
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight



# FILTER FUNCTIONS
def filt(nodes, identifier): 
    mask = ''
    for idx,i in enumerate(identifier):
        mask+='(nodes.data["node_type"]==' + str(i) +')'
        if idx != len(identifier)-1:
            mask+= '|'
    return (eval(mask))#.squeeze(1)
