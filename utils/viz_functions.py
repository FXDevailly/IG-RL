import networkx as nx
import matplotlib.pyplot as plt
def draw_dgl_net(net, file = None):
    g = net.to_networkx()
    pos = nx.kamada_kawai_layout(g)
    plt.figure(3,figsize=(20,20)) 
    nx.draw(g, pos, with_labels=True, node_color=list(net.ndata['node_type']))
    if file is not None:
        plt.savefig(file)
    else:
        plt.show()
        plt.show()
    plt.clf()
    
