import pandas as pd
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import utm
import random
from math import sqrt, log
import geopandas as gpd

# Required Libraries
from tqdm import tqdm as tqdmBasic
from alive_progress import alive_bar

import GeosocialAlgorithms as ga


def generate_colors(n):
    cmap = plt.colormaps['hsv']  # Use o colormap 'hsv' para cores distintas
    colors = cmap(np.linspace(0, 1, n+1))  # Gera n cores distintas
    return colors

def show_graph_metrics(graph):
    """Mostra as métricas de um grafo_twitter.

    Args:
        graph: Grafo a ser analisado.
    """
    degrees = []

    for node in graph.nodes():
        degrees.append(nx.degree(graph, node))
        
    print(f"Nº de nós: {graph.number_of_nodes()}")
    print(f"Nº de links: {graph.number_of_edges()}")
    print(f"Grau médio: {np.mean(degrees)}")
    print(f"Densidade: {nx.density(graph)}")
    print(f"Cluster global: {nx.transitivity(graph)}")
    print(f"Cluster médio: {nx.average_clustering(graph)}")
    
def random_subgraph_by_porcentage(graph: nx.DiGraph, porcentage: float = 0.1, connected_only: bool = False):
    """Gera um subgrafo aleatório de um grafo.

    Args:
        graph (nx.Graph): Grafo base.
        porcentage (float, optional): Porcentagem de subgrafo. Defaults to 0.1.
    """
    graph_return = nx.DiGraph()
    
    edges = random.sample(list(graph.edges()), int(graph.number_of_edges()*porcentage))
    graph_return.add_edges_from(edges)
    
    for node in graph_return.nodes():
        graph_return.nodes[node].update(graph.nodes[node])
        
    if connected_only:
        components = nx.weakly_connected_components(graph_return)
        biggest_component = max(components, key=len)
        connected_graph = graph_return.subgraph(biggest_component)
        return connected_graph
    
    return graph_return

def calculate_distance(x1, y1, x2, y2):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def convert_geo_to_utm(graph: nx.graph):
    """Converte as coordenadas geográficas de um grafo para UTM.

    Args:
        graph (nx.graph): Grafo a ser convertido.

    Returns:
        nx.graph: Grafo convertido.
    """
    for node in graph.nodes():
        latitute = float(graph.nodes[node]['median_lat'])
        longitude = float(graph.nodes[node]['median_lon'])
        
        easting, northing, _, _ = utm.from_latlon(latitute, longitude)
        graph.nodes[node]['median_X'] = easting
        graph.nodes[node]['median_Y'] = northing
        
    return graph

def build_sets_within_2km(graph, x_attr='longitude', y_attr='latitude'):
    nodes = list(graph.nodes())
    sets_within_2km = []

    # Extrair coordenadas projetadas dos nós
    coords = np.array([[graph.nodes[node][x_attr], graph.nodes[node][y_attr]] for node in nodes])

    # Calcular a matriz de distâncias euclidianas
    dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)

    for i, node1 in enumerate(nodes):
        set_within_2km = {node1}
        for j, node2 in enumerate(nodes):
            if i != j and dist_matrix[i, j] <= 2000:
                set_within_2km.add(node2)
        sets_within_2km.append(set_within_2km)

    # Remover subconjuntos e duplicatas
    unique_sets = set()
    for s in sets_within_2km:
        unique_sets.add(frozenset(s))
    
    # Convert back to list of sets
    unique_sets = [set(s) for s in unique_sets]
    
    return unique_sets


def plot_colored_communities(latitude, longitude, grafo, grafo_geo, comunities_geo, comunities_original, with_labels=False):
    pos = {node: (grafo.nodes[node][latitude], grafo.nodes[node][longitude]) for node in grafo.nodes()}
    pos_geo = {node: (grafo_geo.nodes[node][latitude], grafo_geo.nodes[node][longitude]) for node in grafo_geo.nodes()}
    
    colors_original = generate_colors(len(comunities_original))
    colors_geo = generate_colors(len(comunities_geo))
    
    node_colors_original = {}
    node_colors_geo = {}
    
    for i, com in enumerate(comunities_original):
        for node in com:
            node_colors_original[node] = colors_original[i]
            
    for i, com in enumerate(comunities_geo):
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

    plt.show()

if __name__ == "__main__":
    escolha = input("Deseja carregar o grafo teste básico? (s/n): ")
    
    if escolha == 's':
        # Cria um grafo vazio
        grafo = nx.Graph()

        # Adiciona 5 nós ao grafo com atributos de coordenadas euclidianas
        for i in range(4):
            x = random.uniform(0, 10)  # Coordenada x aleatória entre 0 e 10
            y = random.uniform(0, 10)  # Coordenada y aleatória entre 0 e 10
            grafo.add_node(i, x=x, y=y)
            
        grafo.add_node(4, x=100, y=100)
        for i in range(4):
            grafo.add_edge(i, 4)
        
        latitude = 'y'
        longitude = 'x'
        
            
        # # Exibe os atributos dos nós
        # for node in grafo.nodes():
        #     print(f"Node {node}: {grafo.nodes[node]}")
    
    else: 
        grafo_twitter = nx.read_graphml("D:\\Documentos\\data_and_code\\all_data_lisbon\\grafo_twitter_com_coordenadas.graphml")
        
        print("Métricas do grafo original:")
        show_graph_metrics(grafo_twitter)
        print("--------------------")
        
        nodes_without_coordinates = [node for node in grafo_twitter.nodes() if 'median_lat' not in grafo_twitter.nodes[node]]
        
        grafo_twitter.remove_nodes_from(nodes_without_coordinates)
        
        print("Métricas do grafo original sem nós sem coordenadas:")    
        show_graph_metrics(grafo_twitter)
        print("--------------------")
        
        # latitude = input("Digite o nome do atributo de latitude: ")
        # longitude = input("Digite o nome do atributo de longitude: ")
        
        latitude = 'median_Y'
        longitude = 'median_X'
        print("--------------------")

        # df_performance = pd.DataFrame()

        # try:
        #     df_performance = pd.read_csv('E://Comparação entre comunidades - 5.csv', encoding='utf-8')
        # except FileNotFoundError:
        #     df_performance = pd.DataFrame(columns=["Geomodularidade Louvain", "Geomodularidade Conjuntos"])
            
        # times_to_run = 100
        # with alive_bar(times_to_run, bar='blocks') as bar:
        #     for _ in range(times_to_run):
        
        reduzir = input("Deseja reduzir o grafo? (s/n): ")
        
        if reduzir == 's':
            grafo = random_subgraph_by_porcentage(grafo_twitter, porcentage=0.01, connected_only=True)    
        else:
            grafo = grafo_twitter.copy()
            
        print("Métricas do grafo reduzido:")
        show_graph_metrics(grafo) 
        print("--------------------")
        
        grafo = convert_geo_to_utm(grafo)

                                
        # comunidades_algoritmo = build_sets_within_2km(grafo, x_attr=longitude, y_attr=latitude)
    
    # comunities = nx.community.louvain_communities(grafo)
    # print(f"Comunidades: {comunities}")
    # print(f"Geomodularidade das comunidades de Louvain: {geosocial.geo_social_modularity(communities=comunities, d=d)}")
    # print("--------------------")
    # # print(f"Conjuntos de nós: {sets_within_2km}")
    
    # geo_louvain = geosocial.geo_social_modularity(communities=comunities, d=d)
    
    # comunidades_algoritmo = geosocial.custom_naive_greedy_modularity_communities(d=d)
    
    # print(f"Geomodularidade das pseudo-comunidades geográficas: {geosocial.geo_social_modularity(comunidades_algoritmo, d=d)}")
    # print("--------------------")
    
    # geo_sets = geosocial.geo_social_modularity(comunidades_algoritmo, d=d)
    
    # print("Comunidades do algoritmo:")
    # print(comunidades_algoritmo)
    
    # print(f"Média de nós por comunidade: {np.mean([len(c) for c in comunidades_algoritmo])}")
    
    # df_performance = df_performance._append({"Geomodularidade Louvain": geo_louvain, "Geomodularidade Conjuntos": geo_sets}, ignore_index=True)
    
    # del grafo, sets_within_2km, geosocial, comunities, geo_louvain, geo_sets
    
    # bar()
        
    # df_performance.to_csv('E://Comparação entre comunidades - 5.csv', encoding='utf-8')
    # print("Fim do teste.")
    
    # plot_colored_communities(generate_colors, latitude, longitude, grafo, comunidades_algoritmo, comunities)
    
    print("Métricas do grafo original:")
    show_graph_metrics(grafo)
    print("--------------------")
    
    # Add new attribute to each node representing a tuple of lat and long
    try:
        for node in grafo.nodes():
            lat = grafo.nodes[node]['median_Y']
            lon = grafo.nodes[node]['median_X']
            grafo.nodes[node]['coords'] = (lat, lon)
                        
    except KeyError:
        for node in grafo.nodes():
            lat = grafo.nodes[node]['y']
            lon = grafo.nodes[node]['x']
            grafo.nodes[node]['coords'] = (lat, lon)
            
    
    pos = {node: (grafo.nodes[node][latitude], grafo.nodes[node][longitude]) for node in grafo.nodes()}
    
    # # Plot the graphs side by side
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # nx.draw_networkx(grafo, pos=pos, node_size=100, font_size=10, with_labels=True)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.xlabel('x', fontsize=12)
    # plt.ylabel('y', fontsize=12)
    # plt.axis('on')
    # plt.title("Original Graph")    
    # plt.show()
        
    geosocial = ga.GeoSocial(grafo, lat=latitude, lon=longitude)
    
    geosocial.indexing_graph()
    
    grafo_geo = geosocial.return_geographic_graph_by_radius(10000, coords_str='coords')
    
    print("Métricas do grafo geográfico:")
    show_graph_metrics(grafo_geo)
    print("--------------------")
        
    # print("Atributos dos nós do grafo original:")
    # for node in grafo.nodes():
    #     attributes = grafo.nodes[node]
    #     print(f"Node: {node}")
    #     print(f"Attributes: {attributes}")
    #     print("--------------------")
        
    # print("Atributos dos nós do grafo geográfico:")
    # for node in grafo_geo.nodes():
    #     attributes = grafo_geo.nodes[node]
    #     print(f"Node: {node}")
    #     print(f"Attributes: {attributes}")
    #     print("--------------------")
        
    # pos_original = {node: (grafo.nodes[node][latitude], grafo.nodes[node][longitude]) for node in grafo.nodes()}
    # pos_geo = {node: (grafo_geo.nodes[node][latitude], grafo_geo.nodes[node][longitude]) for node in grafo_geo.nodes()}   
    
    # # Plot the graphs side by side
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # nx.draw_networkx(grafo, pos=pos_original, node_size=100, font_size=10, with_labels=False)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.xlabel('x', fontsize=12)
    # plt.ylabel('y', fontsize=12)
    # plt.axis('on')
    # plt.title("Original Graph")

    # plt.subplot(1, 2, 2)
    # nx.draw_networkx(grafo_geo, pos=pos_geo, node_size=100, font_size=10, with_labels=False) 
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.xlabel('x', fontsize=12)
    # plt.ylabel('y', fontsize=12)
    # plt.axis('on')
    # plt.title("Geographic Graph")

    # plt.show()
    
    comunidades_social = nx.community.greedy_modularity_communities(grafo)
    comunidades_geo = nx.community.greedy_modularity_communities(grafo_geo)
    
    plot_colored_communities(latitude, longitude, grafo, grafo_geo, comunidades_geo, comunidades_social, with_labels=False)