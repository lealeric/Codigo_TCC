"""
GeoSocial Module

This module is used to perform geosocial analysis on social media data.
"""
import numpy as np
import math
import heapq
import networkx as nx
import utm
from math import log
from scipy.cluster.hierarchy import ward
from itertools import combinations

class GeoSocial():
    """
    GeoSocial class for performing geosocial analysis on social media data.
    """
    
    def __init__(self, graph: nx.Graph, D = None, lat:str = "latitude", lon:str = "longitude"):
        """
        Initializes the GeoSocial class.
        
        Parameters:
        graph: nx.Graph
            The graph object.
        lat: str
            The name of the column containing the latitude values.
        lon: str
            The name of the column containing the longitude values.                        
        """        
        if graph is None:
            raise ValueError("You must provide a Networkx graph.")
        
        self.graph = graph
        
        # Calculate the geographic centroid
        
        self.latitudes = [float(self.graph.nodes[node][lat]) for node in self.graph.nodes if lat in self.graph.nodes[node]]
        self.longitudes = [float(self.graph.nodes[node][lon]) for node in self.graph.nodes if lon in self.graph.nodes[node]]
        
        self.latitude = lat
        self.longitude = lon
        
        self.centroid = self.calculate_geographic_centroid()
        
        self.D = D 
        
        self.find_general_diameter()
        
    
    def create_graph_internal(self, data):
        """
        Creates a graph from a pandas DataFrame.
        """
        self.graph = nx.from_pandas_edgelist(data)
        
    def structure_modularity(self):
        """
        Returns the modularity of the graph.
        """
        return nx.community.modularity(self.graph)
    
           
    def find_general_diameter(self, round_to: int = 0):
        """
        Finds the two most distant points in the graph.
        """
        max_lat = float('-inf')
        min_lat = float('inf')
        max_lon = float('-inf')
        min_lon = float('inf')

        for lat, lon in zip(self.latitudes, self.longitudes):
            if lat > max_lat:
                max_lat = lat
            if lat < min_lat:
                min_lat = lat
            if lon > max_lon:
                max_lon = lon
            if lon < min_lon:
                min_lon = lon
                
        def calculate_distance_projected(x1, y1, x2, y2):
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        def calculate_distance_geographic(lat1, lon1, lat2, lon2):
            
            x1, y1, _, _ = utm.from_latlon(lat1, lon1)
            x2, y2, _, _ = utm.from_latlon(lat2, lon2)
            
            return calculate_distance_projected(x1, y1, x2, y2)

        # if abs(max_lon) > 180:
        self.D = round(calculate_distance_projected(max_lat, max_lon, min_lat, min_lon), round_to)
        # else:
        #     self.D = round(calculate_distance_geographic(max_lat, max_lon, min_lat, min_lon), round_to)
            
    def calculate_geographic_centroid(self):
        """
        Calculates the geographic centroid of the graph.
        """
        return (np.mean(self.latitudes), np.mean(self.longitudes))
    
    def indexing_graph(self):
        """
        Indexes the graph.
        """
        self.graph = nx.convert_node_labels_to_integers(self.graph, label_attribute="username")
    
    def _geometric_edges(self, radius: float, p, pos_name):
        """
        Adds edges to the graph based on a radius.
        """
        nodes_pos = self.graph.nodes(data=pos_name)        
        try:
            import scipy as sp
        except ImportError:
            # no scipy KDTree so compute by for-loop
            radius_p = radius**p
            edges = [
                (u, v)
                for (u, pu), (v, pv) in combinations(nodes_pos, 2)
                if sum(abs(a - b) ** p for a, b in zip(pu, pv)) <= radius_p
            ]
            return edges
        # scipy KDTree is available
        nodes, coords = list(zip(*nodes_pos))
        kdtree = sp.spatial.cKDTree(coords)  # Cannot provide generator.
        edge_indexes = kdtree.query_pairs(radius, p)
        edges = [(nodes[u], nodes[v]) for u, v in sorted(edge_indexes)]
        return edges
    
    def return_geographic_graph_by_radius(self, radius: float, p: int = 2, coords_str: str = 'coords', us_metric: bool = False) -> nx.Graph:
        """
        Returns a geographic graph based on a radius.
        
        Args:
            radius (float): The radius of the circle in meters.
            p (int): The power parameter for the Minkowski distance.
            coords_str (str): The name of the attribute containing the coordinates in projected space.
            us_metric (bool): If True, the radius is in miles. The default is False.
            
        Raises:
            ValueError: If the graph nodes do not have coordinates.            
            
        Returns:
            nx.Graph: The geographic graph.
            
        Examples:
            >>> G = nx.Graph()
            >>> G.add_node(1, coords=(0, 0))
            >>> G.add_node(2, coords=(1, 1))
            >>> G.add_node(3, coords=(2, 2))
            >>> gs = GeoSocial(G)
            >>> G_geo = gs.return_geographic_graph_by_radius(1)
            >>> G_geo.nodes(data=True)
            NodeDataView({1: {'coords': (0, 0)}, 2: {'coords': (1, 1)}})
            >>> G_geo.edges()
            EdgeView([(1, 2)])
        """
        if us_metric:
            radius = radius * 0.000621371
        
        index_changed = False
        
        if not isinstance(list(self.graph.nodes)[0], int):
            index_changed = True
            self.indexing_graph()        
                
        if not all(coords_str in self.graph.nodes[node] for node in self.graph.nodes):
            raise ValueError("You must provide coordinates in the graph nodes.")                
                
        # G = nx.random_geometric_graph(self.graph.number_of_nodes(), radius, pos=self.graph.nodes(data=coords_str), p=p)       
        
        G = nx.empty_graph(self.graph.number_of_nodes())
        
        G.add_nodes_from(self.graph.nodes(data=True))

        G.add_edges_from(self._geometric_edges(radius, p, coords_str))
        
        # Create a mapping dictionary with the new node labels
        if 'username' in self.graph.nodes[list(self.graph.nodes)[0]]:
            mapping = {node: G.nodes[node]['username'] for node in G.nodes()}

            # Relabel the nodes in the graph using the mapping dictionary
            G = nx.relabel_nodes(G, mapping)
        
        return G

    # def calculate_ward_dispersion(self, community):
    #     coordinates = [(float(self.graph.nodes[node]['latitude']), float(self.graph.nodes[node]['longitude'])) for node in community]
    #     distance_matrix = np.zeros((len(coordinates), len(coordinates)))

    #     # Calculate pairwise distances between coordinates
    #     for i in range(len(coordinates)):
    #         for j in range(i+1, len(coordinates)):
    #             distance_matrix[i, j] = self.calculate_distance_projected(coordinates[i][0], coordinates[i][1], coordinates[j][0], coordinates[j][1])
    #             distance_matrix[j, i] = distance_matrix[i, j]

    #     # Perform Ward's method clustering
    #     linkage_matrix = ward(distance_matrix)

    #     # Return the resulting linkage matrix
    #     return linkage_matrix
        
    def geo_social_modularity(self, communities: list, graph: nx.graph = None, d: float = 0.1):
        """
        Calcula a modularidade geo-social de uma rede com base nas comunidades fornecidas.

        Parâmetros:
            communities (list): Uma lista de comunidades, onde cada comunidade é representada como um conjunto de nós.
            graph (nx.graph, opcional): O grafo da rede. Se não for fornecido, o grafo associado ao objeto é usado.
            d (float, opcional): Um parâmetro usado no cálculo da transformação. O padrão é 0.1.

        Retorna:
            float: O valor da modularidade geo-social.

        Raises:
            None

        """
        if graph is None:
            graph = self.graph
        
        def calculate_ward_dispersion(grafo: nx.graph, comunidades):            
            coords = np.array([[float(grafo.nodes[node][self.longitude]), float(grafo.nodes[node][self.latitude])] for node in comunidades])
            
            def community_dispersion(coords: np.array):
                centroid = np.mean(coords, axis=0)
                
                return np.sum((coords - centroid) ** 2)
                    
            return community_dispersion(coords)
        
        def calculate_transformation(dispersion: float, d: float):
            if dispersion == np.inf:
                return np.inf
            return max(-1, min(1, (log(dispersion) + log(self.D) + log(d)) / (log(self.D) - log(d))))
                       
        dispersao = 0
        existe_degeneracao = True
        for i, community in enumerate(communities): 
            if len(community) > 1:
                existe_degeneracao = False
                dispersao_comunidade = calculate_ward_dispersion(graph, community)      
                dispersao += 1/dispersao_comunidade if dispersao_comunidade != 0 else np.inf
        
        if existe_degeneracao:
            return -1
        
        return calculate_transformation(dispersao, d)
    
    @nx._dispatchable(edge_attrs="weight")
    def custom_naive_greedy_modularity_communities(self, resolution=1, d=0.1):
        # First create one community for each node
        communities = [frozenset([u]) for u in self.graph.nodes()]
        # Track merges
        merges = []
        # Greedily merge communities until no improvement is possible
        old_modularity = None
        new_modularity = self.geo_social_modularity(communities, d=d)
        while old_modularity is None or new_modularity > old_modularity:
            # Save modularity for comparison
            old_modularity = new_modularity
            # Find best pair to merge
            trial_communities = list(communities)
            to_merge = None
            for i, u in enumerate(communities):
                for j, v in enumerate(communities):
                    # Skip i==j and empty communities
                    if j <= i or len(u) == 0 or len(v) == 0:
                        continue
                    # Merge communities u and v
                    trial_communities[j] = u | v
                    trial_communities[i] = frozenset([])
                    trial_modularity = self.geo_social_modularity(trial_communities, d=d)
                    if trial_modularity >= new_modularity:
                        # Check if strictly better or tie
                        if trial_modularity > new_modularity:
                            # Found new best, save modularity and group indexes
                            new_modularity = trial_modularity
                            to_merge = (i, j, new_modularity - old_modularity)
                        elif to_merge and min(i, j) < min(to_merge[0], to_merge[1]):
                            # Break ties by choosing pair with lowest min id
                            new_modularity = trial_modularity
                            to_merge = (i, j, new_modularity - old_modularity)
                    # Un-merge
                    trial_communities[i] = u
                    trial_communities[j] = v
            if to_merge is not None:
                # If the best merge improves modularity, use it
                merges.append(to_merge)
                i, j, dq = to_merge
                u, v = communities[i], communities[j]
                communities[j] = u | v
                communities[i] = frozenset([])
        # Remove empty communities and sort
        return sorted((c for c in communities if len(c) > 0), key=len, reverse=True)
