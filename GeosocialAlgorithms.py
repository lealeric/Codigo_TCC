"""
GeoSocial Module

This module is used to perform geosocial analysis on social media data.
"""
import numpy as np
import math
import networkx as nx
import math
import utm

class GeoSocial():
    """
    GeoSocial class for performing geosocial analysis on social media data.
    """
    
    def __init__(self, graph: nx.Graph, lat:str = "latitude", lon:str = "longitude"):
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
        
        self.centroid = self.calculate_geographic_centroid()
        
        self.D = None 
        
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

        if abs(max_lon) > 180:
            self.D = round(calculate_distance_projected(max_lat, max_lon, min_lat, min_lon), round_to)
        else:
            self.D = round(calculate_distance_geographic(max_lat, max_lon, min_lat, min_lon), round_to)
            
    def calculate_geographic_centroid(self):
        """
        Calculates the geographic centroid of the graph.
        """
        return (np.mean(self.latitudes), np.mean(self.longitudes))
        
    def geo_social_modularity(self, d: float = 0.1):
        """
        Returns the modularity of the graph.
        
        Parameters:
        d: float
            The distance between the two most distant points in the graph.
            
        Returns:
        float
            The modularity of the graph.
        """
        
        def geomodularity(x, d):
        
            f_x = (math.log(x) + math.log(self.D) + math.log(d)) / (math.log(self.D) - math.log(d))
            
            return math.log(f_x) 
        
        return sum(map(lambda x: geomodularity(x, d), d))

