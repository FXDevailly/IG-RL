

class Edge():
    def __init__(self, edge_id):


        self.edge_id= edge_id
        self.next_edges = set()
        self.prev_edges = set()
