import networkx as nx
import numpy as np
import random
import math

from networkx.utils import py_random_state
from itertools import combinations

class ControlledWaxmanGraph():
    def __init__(self, graph, k_param, radius):
        self.ALPHAS = {
                10: 0.00068924184549,
                20: 0.0013184686858,
                40: 0.0025755779769,
            }
        
        self.graph = graph
        self.generate_graph(k_param, radius)
        self.nodes = list(self.graph.nodes())
        
        self.original_edges = [random.choice([(u,v), (v,u)]) 
                               for (u,v) in self.graph.edges() 
                               if u < v]
        
        random.shuffle(self.original_edges)
        
    def add_coords(self, G: nx.Graph, radius: int = 1) -> nx.Graph:
        """Add x and y coordinates to each node in the graph.
        
        Args:
            G (nx.Graph): A graph.
            radius (int, optional): Radius of the circle. Defaults to 1.
        
        Returns:
            nx.Graph: Returns a graph with x and y coordinates.
        """
        angle_increment = 2 * np.pi / G.number_of_nodes()
    
        # Assign x and y coordinates to each node
        for i, node in enumerate(G.nodes()):
            angle = i * angle_increment
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            G.nodes[node]['x'] = x
            G.nodes[node]['y'] = y
            G.nodes[node]['coords'] = (x, y)
        
        return G
            
    def calculate_distance(self, n:int, k:int, radius:float) -> float:
        """Calculate the distance between two nodes.
        
        Args:
            n (int): Number of nodes
            k (int): Parameter k of linking edges.
            radius (float): Radius of the circle.
        
        Returns:
            float: Returns the distance between two nodes.
        """
        a = 1
        b = 1
        angle = math.pi * k / n

        distance = math.sqrt(a**2 + b**2 - 2*a*b*math.cos(angle))*radius
        
        return distance
        
    @py_random_state(3)
    def generate_graph(self, k: int, radius: int, seed=None) -> nx.Graph:
        """Generate a Waxman graph.
        
        Args:
            k(int): Parameter k of linking edges.
            radius(int): Radius of the circle.
            seed (optional): Random seed. Defaults to None            
        
        Returns:
            nx.Graph: Returns a Waxman graph.
            
        Raises:
            nx.NetworkXError: If k <= 0, a NetworkXError is raised.
            
        """
        if k <= 0:
            raise nx.NetworkXError("k must be positive")
        
        n = self.graph.number_of_nodes()
                
        distance = self.calculate_distance(n, k, radius)
        
        beta_waxman = 1
        l_waxman = radius*2
        # alpha_waxman = distance / (2 * math.log(2))
        alpha_waxman = self.ALPHAS[k]
        
        def dist(u, v):
            x1, y1 = self.graph.nodes[u]['coords']
            x2, y2 = self.graph.nodes[v]['coords']
            return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # `pair` is the pair of nodes to decide whether to join.b
        def should_join(pair):
            u, v = pair
            return seed.random() < beta_waxman * math.exp(-dist(u, v) / (self.ALPHAS[k] * 2))

        print("Generating edges...")
        print(f"Alpha: {alpha_waxman}")
        self.graph.add_edges_from(filter(should_join, combinations(self.graph, 2)))
        
        return self.graph
    
    @py_random_state(2)
    def randomize_edges(self, e, seed=None) -> None:
        """Randomize e edges of the graph.

        Args:
            e: Number of edges to be randomized.
            seed (optional): Random seed. Defaults to None.

        Raises:
            nx.NetworkXError: number of edges to be randomized is greater than the number of original edges.
        """
        if e > len(self.original_edges):
            raise nx.NetworkXError("e > original_edges, choose smaller e")
        
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
        