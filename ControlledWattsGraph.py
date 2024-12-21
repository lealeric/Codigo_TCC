import networkx as nx
import random

class ControlledWattsGraph():
        
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.nodes = list(range(n))
        self.graph = self.generate_graph()
        
        self.original_edges = [random.choice([(u,v), (v,u)]) 
                               for (u,v) in self.graph.edges() 
                               if u < v]
        
        random.shuffle(self.original_edges)
        
    def generate_graph(self) -> nx.Graph:
        """Generate a Watts-Strogatz small-world graph.

        Raises:
            nx.NetworkXError: If k > n, a NetworkXError is raised.

        Returns:
            nx.Graph: Returns a Watts-Strogatz small-world graph.
        """
        if self.k > self.n:
            raise nx.NetworkXError("k>n, choose smaller k or larger n")

        # If k == n, the graph is complete not Watts-Strogatz
        if self.k == self.n:
            return nx.complete_graph(self.n)

        G = nx.Graph()
        # connect each node to k/2 neighbors
        for j in range(1, self.k // 2 + 1):
            targets = self.nodes[j:] + self.nodes[0:j]  # first j nodes are now last in list
            G.add_edges_from(zip(self.nodes, targets))
                    
        return G
    
    def randomize_edges(self, e, seed=None) -> None:
        """Randomize e edges of the graph.

        Args:
            e: Number of edges to be randomized.
            seed (optional): Random seed. Defaults to None.

        Raises:
            nx.NetworkXError: _description_
        """
        if e > len(self.original_edges):
            raise nx.NetworkXError("e > original_edges, choose smaller e")
        
        if seed is None:
            seed = random.Random()
        
        for _ in range(e):
            (u,v) = self.original_edges.pop()
            
            while True:
                w = seed.choice(self.nodes)
                
                if (u,w) in self.graph.edges() \
                    or u == w:
                    continue
                
                self.graph.remove_edge(u, v)
                self.graph.add_edge(u, w)
                break