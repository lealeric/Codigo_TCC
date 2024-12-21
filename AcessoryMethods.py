import csv
import json
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import utm
from tqdm import tqdm as tqdmBasic


class AcessoryMethods:
    """
        Acessory methods for the project.
    """
    def __init__(self):
        """
            Constructor method.
        """
        pass
    
    
    def calculate_distance_geographic(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the distance between two points in the geographic space.
        
        Args:
            lat1 (float): Latitude of the first point.
            lon1 (float): Longitude of the first point.
            lat2 (float): Latitude of the second point.
            lon2 (float): Longitude of the second point.
            
        Returns:
            float: The distance between the two points.
            
        Example:
            >>> calculate_distance_geographic(37.7749, -122.4194, 34.0522, -118.2437)
            559.23
        """
        x1, y1, _, _ = utm.from_latlon(lat1, lon1)
        x2, y2, _, _ = utm.from_latlon(lat2, lon2)

        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    
    def calculate_distance(self, xp1: float, yp1: float, xp2: float, yp2: float) -> float:
        """
        Calculate the distance between two points in the Euclidean space.
        
        Args:
            xP1 (float): X-coordinate of the first point.
            yP1 (float): Y-coordinate of the first point.
            xP2 (float): X-coordinate of the second point.
            yP2 (float): Y-coordinate of the second point.
            
        Returns:
            float: The distance between the two points.
            
        Example:
            >>> calculate_distance(0, 0, 3, 4)
            5.0
        """
        p1 = [xp1, yp1]
        p2 = [xp2, yp2]
        
        return math.dist(p1, p2)

    def generate_colors(self, n: int) -> np.ndarray:
        """
        Generate a list of colors for plotting.
        
        Args:
            n (int): Number of colors to generate.
            
        Returns:
            np.ndarray: A list of colors.
            
        Example:
            >>> generate_colors(5)
            array([[1.        , 0.        , 0.        , 1.        ],
                [1.        , 0.5       , 0.        , 1.        ],
                [1.        , 1.        , 0.        , 1.        ],
                [0.5       , 1.        , 0.        , 1.        ],
                [0.        , 1.        , 0.        , 1.        ]])
        """
        cmap = plt.colormaps['hsv']
        colors = cmap(np.linspace(0, 1, n+1))
        return colors

    
    def convert_geo_to_utm(self, graph: nx.Graph, column_lat: str = 'median_lat', column_lon: str = 'median_lon') -> nx.Graph:
        """
        Convert the geographic coordinates of the nodes in the graph to UTM coordinates.

        Args:
            graph (nx.Graph): A graph object.
            column_lat (str, optional): Column name of the latitude component. Defaults to 'median_lat'.
            column_lon (str, optional): Column name of the longitude component. Defaults to 'median_lon'.

        Returns:
            nx.Graph: A graph object with UTM coordinates.
            
        Example:
            >>> G = nx.Graph()
            >>> G.add_node(1, median_lat=37.7749, median_lon=-122.4194)
            >>> G.add_node(2, median_lat=34.0522, median_lon=-118.2437)
            >>> convert_geo_to_utm(G)
            >>> print(G.nodes[1]['median_X'], G.nodes[1]['median_Y'])
            551730.0 4182689.0                
        """
        for node in graph.nodes():
            latitute = float(graph.nodes[node][column_lat])
            longitude = float(graph.nodes[node][column_lon])
            
            easting, northing, _, _ = utm.from_latlon(latitute, longitude)
            graph.nodes[node]['median_X'] = easting
            graph.nodes[node]['median_Y'] = northing
            
        return graph

    
    def show_graph_metrics(self, graph: nx.Graph, clusters: bool = False) -> None:
        """
        Show some metrics of the graph.
        
        Args:
            graph (nx.graph): A graph object.
            clusters (bool, optional): If the clusters should be displayed. Defaults to False.
            
        Example:
            >>> G = nx.Graph()
            >>> G.add_node(1)
            >>> G.add_node(2)
            >>> G.add_edge(1, 2)
            >>> show_graph_metrics(G)
            Nº de nós: 2
            Nº de links: 1
            Grau médio: 1.0
            Densidade: 1.0
        """    
        degrees = []

        for node in graph.nodes():
            degrees.append(nx.degree(graph, node))
            
        print(f"Nº de nós: {graph.number_of_nodes()}")
        print(f"Nº de links: {graph.number_of_edges()}")
        print(f"Grau médio: {np.mean(degrees)}")
        print(f"Densidade: {nx.density(graph)}")
        
        if clusters:
            print(f"Cluster global: {nx.transitivity(graph)}")
            print(f"Cluster médio: {nx.average_clustering(graph)}")

    
    def return_graph_metrics(self, graph: nx.Graph) -> dict:
        """
        Return some metrics of the graph.

        Args:
            graph (nx.graph): A graph object.

        Returns:
            dict: A dictionary with the metrics.
            
        Example:
            >>> G = nx.Graph()
            >>> G.add_node(1)
            >>> G.add_node(2)
            >>> G.add_edge(1, 2)
            >>> return_graph_metrics(G)
            {'numero_nos': 2, 'numero_links': 1, 'grau_medio': 1.0, 'densidade': 1.0}
        """
        degrees = []

        for node in graph.nodes():
            degrees.append(nx.degree(graph, node))
            
        return {
            "numero_nos": graph.number_of_nodes(),
            "numero_links": graph.number_of_edges(),
            "grau_medio": np.mean(degrees),
            "densidade": nx.density(graph)
        }

    
    def merge_duplicate_nodes(self, graph: nx.Graph) -> tuple[nx.Graph, dict]:
        """Merge duplicate nodes in a graph.

        Args:
            graph (nx.Graph): Graph to be processed.

        Raises:
            KeyError: If the graph does not have a 'coords' attribute in the nodes.
            
        Returns:
            tuple: Tupla contendo:
                graph (nx.Graph): Graph with merged nodes.
                unique_coords (dict): Dictionary with the unique coordinates.
            
        Examples:
            >>> G = nx.Graph()
            >>> G.add_node(1, coords=(1, 2))
            >>> G.add_node(2, coords=(1, 2))
            >>> G.add_node(3, coords=(3, 4))
            
            >>> G.add_edge(1, 2)
            >>> G.add_edge(2, 3)
            
            >>> G, unique_coords = merge_duplicate_nodes(G)
            >>> print(unique_coords)
            {(1, 2): 1, (3, 4): 3}
        """
        
        unique_coords = {}

        for node in list(graph.nodes()):
            try:
                coords = graph.nodes[node]['coords']
            except KeyError as exc:
                raise KeyError("The graph must have a 'coords' attribute in the nodes.") from exc
            
            if coords in unique_coords:      
                graph.add_edges_from(graph.edges(node))        
                graph.remove_node(node)
            else:        
                unique_coords[coords] = node
                
        return graph, unique_coords

    
    def map_community_nodes(self, community: list) -> dict:
        """Atribui um número de comunidade a cada nó de cada grafo.

        Args:
            community (list): Lista de comunidades do grafo 1.

        Returns:
            node_community_map (dict): Dicionário com o mapeamento dos nós para as comunidades.
            
        Examples:
            >>> community1 = [{1, 2, 3}, {4, 5, 6}]
            >>> map_community_nodes(community1)
            {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
        """
        node_community_map = {}

        for i, comm in enumerate(community):
            for node in comm:
                node_community_map[node] = i
                
        return node_community_map

    
    def jaccard_similarity(self, community1: set, community2: set) -> float:
        """Calcula a similaridade de Jaccard entre duas comunidades.

        Args:
            community1 (set): Comunidade 1.
            community2 (set): Comunidade 2.

        Returns:
            float: O índice de similaridade de Jaccard.
            
        Examples:
            >>> community1 = {1, 2, 3}
            >>> community2 = {2, 3, 4}
            >>> jaccard_similarity(community1, community2)
            0.5
        """
        intersection = len(community1.intersection(community2))
        union = len(community1.union(community2))
        
        return intersection / union

    
    def assign_labels(self, partition, all_elements):
        """Define os rótulos de cada elemento com base na partição.

        Args:
            partition (list): A partição.
            all_elements (list): Todos os elementos.
            
        Returns:
            list: Uma lista com os rótulos de cada elemento.
            
        Examples:
            >>> partition = [{1, 2, 3}, {4, 5, 6}]
            >>> all_elements = [1, 2, 3, 4, 5, 6]
            >>> assign_labels(partition, all_elements)
            [0, 0, 0, 1, 1, 1]
        """
        labels = {}
        for cluster_id, cluster in enumerate(partition):
            for element in cluster:
                labels[element] = cluster_id
        return [labels[element] for element in all_elements]

    
    def plot_scatter_empyrical_complementar_distribution(self, distances, output_path: str = 'E://', output_file_name: str = 'empyrical_complementar_distribution.png', show: bool = False, log: bool = False):
        """Plota o gráfico de dispersão da distribuição empírica complementar.

        Args:
            distances (list): Lista com as distâncias.
            output_path (str): Caminho de saída do arquivo.
            output_file_name (str): Nome do arquivo de saída.
            show (bool): Se o gráfico deve ser exibido.                
        """    
        # sort the distances

        # calculate the probability of each distance
        if isinstance(distances, dict):
            keys_in_order = sorted(distances.keys())
            length_distances = sum(distances.values())
            dist = 1
            prob = []
            print(f"keys_in_order: {keys_in_order}")
            print(f"length_distances: {length_distances}")
            
            distances_list = [key for key, value in distances.items() for _ in tqdmBasic(range(value))]
            n = len(distances_list)
            prob = [1 - (i+1)/n for i in tqdmBasic(range(0,n))]
                        
            distances = list(distances.keys())
            
        else:
            distances.sort()
            if log == True:
                prob = [np.log10(1 - (i/len(distances))) for i in tqdmBasic(range(len(distances)), desc="Calculando probabilidade")]
            else:
                prob = [1 - (i/len(distances)) for i in tqdmBasic(range(len(distances)))]
                
        print(type(prob))

        # plot the scatter graph
        plt.figure(figsize=(180, 120))
        plt.scatter(distances, prob)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel("Distância")
        plt.ylabel("Probabilidade")
        plt.savefig(f"{output_path}{output_file_name}", dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()

        plt.close()
    
    def plot_colored_communities(self, grafo: nx.Graph, grafo_geo: nx.Graph, communities_social: list, communities_geo: list, 
                                latitude: str = 'lat', longitude: str = 'long', with_labels: bool = False, use_geolocation: bool = True,
                                output_path: str = 'E://', output_file_name: str = 'comunidades.png'):
        """
        Plot the graph with colored nodes based on communities.

        Args:
            grafo (nx.Graph): A graph object.
            grafo_geo (nx.Graph): A graph object with geographic coordinates.
            communities_social (list): A list of communities of the social graph.
            communities_geo (list): A list of communities of the geographic graph.
            latitude (str, optional): Column name of the latitude component. Defaults to 'lat'.
            longitude (str, optional): Column name of the longitude component. Defaults to 'long'.
            with_labels (bool, optional): If the labels should be displayed. Defaults to False.
            use_geolocation (bool, optional): If the geographic coordinates should be used. Defaults to True.
        """        
        if use_geolocation:
            pos = {node: (grafo.nodes[node][latitude], grafo.nodes[node][longitude]) for node in grafo.nodes()}
        else:
            pos = nx.spring_layout(grafo)
            
        pos_geo = {node: (grafo_geo.nodes[node][latitude], grafo_geo.nodes[node][longitude]) for node in grafo_geo.nodes()}
        
        colors_original = self.generate_colors(len(communities_social))
        colors_geo = self.generate_colors(len(communities_geo))
        
        node_colors_original = {}
        node_colors_geo = {}
        
        for i, com in enumerate(communities_social):
            for node in com:
                node_colors_original[node] = colors_original[i]
                
        for i, com in enumerate(communities_geo):
            for node in com:
                node_colors_geo[node] = colors_geo[i]    

        # Plot the graph with colored nodes based on communities
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 1)
        nx.draw_networkx(grafo, pos=pos, node_color=[node_colors_original[node] for node in grafo.nodes()], node_size=100, font_size=10, with_labels=with_labels)
        plt.axis('on')
        plt.title("Comunidades do grafo social")

        plt.subplot(1, 2, 2)
        nx.draw_networkx(grafo_geo, pos=pos_geo, node_color=[node_colors_geo[node] for node in grafo_geo.nodes()], node_size=100, font_size=10, with_labels=with_labels)
        plt.axis('on')
        plt.title("Comunidades do grafo geográfico")

        # plt.show()
        plt.savefig(f"{output_path}{output_file_name}", dpi=300, bbox_inches='tight')
        plt.close()

    
    def _export_dictionary(self, data_dict: dict, path : str = 'E://', file_name : str = 'dict.json'):
        """Exporta um dicionário para um arquivo JSON

        Args:
            dict (dict): dicionário a ser exportado
            path (str, optional): Caminho do arquivo de saída. Defaults to 'E:/'.
            file_name (str, optional): Nome do arquivo de saída. Defaults to 'dict.json'.
        """                
        with open(f"{path}{file_name}", 'w', encoding='utf-8') as f:
            json.dump(data_dict, f)

    
    def _export_list_to_csv(self, data_list: list, path : str = 'E://', file_name : str = 'list.csv'):
        """Exporta uma lista para um arquivo CSV

        Args:
            data_list (list): Lista a ser exportada
            path (str, optional): Caminho do arquivo de saída. Defaults to 'E:/'.
            file_name (str, optional): Nome do arquivo de saída. Defaults to 'list.csv'.
        """                
        with open(f"{path}{file_name}", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data_list)

    
    def _export_graphml(self, graph : nx.Graph, path : str = 'E://', file_name : str = 'graph.graphml'):
        """Exporta um grafo para um arquivo GraphML

        Args:
            graph (nx.Graph): Grafo a ser exportado
            path (str, optional): Caminho do arquivo de saída. Defaults to 'E:/'.
            file_name (str, optional): Nome do arquivo de saída. Defaults to 'graph.graphml'.
        """
        
        # remove the 'coords' attribute from the nodes
        
        for node in graph.nodes():
            if 'coords' in graph.nodes[node]:
                del graph.nodes[node]['coords']
            if 'pos' in graph.nodes[node]:
                del graph.nodes[node]['pos']
        
        nx.write_graphml(graph, f"{path}{file_name}")

    
    def print_progress_bar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', print_end = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = print_end)
        # Print New Line on Complete
        if iteration == total: 
            print()