import pandas as pd
import numpy as np
import random
import time

import networkx as nx
import math

import os



grafo_twitter = nx.read_graphml("D:\\Documentos\\data_and_code\\all_data_lisbon\\grafo_twitter.graphml")
grafo_twitter_conexo = nx.read_graphml("D:\\Documentos\\data_and_code\\all_data_lisbon\\grafo_twitter_conexo.graphml")

def show_graph_metrics(graph: nx.graph):
    """Mostra as métricas de um grafo.

    Args:
        graph (nx.graph): Grafo a ser analisado.
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
    
def calculate_geostatistics(graph: nx.graph = None, df: pd.DataFrame = None, column: str = 'comunidade') -> dict:
    """Calcula as estatísticas geográficas de um grafo.

    Args:
        graph (nx.graph): Grafo a ser analisado.
        

    Returns:
        dict: Dicionário com as estatísticas calculadas.
        
    Raises:
        ValueError: Se não for fornecido um grafo ou um dataframe.
    """
    stats = {}
    
    if graph is not None:
        for node in graph.nodes():
            community = graph.nodes[node][column]
            if community not in stats.keys():
                stats[community] = {
                    'x_points': [],
                    'y_points': [],
                    'x_mean': 0.0,
                    'y_mean': 0.0,
                    'mean_distance_to_center': 0.0,
                    'standard_deviation': 0.0,
                }
                
            if 'median_X' not in graph.nodes[node]:
                continue
            
            x = float(graph.nodes[node]['median_X'])
            y = float(graph.nodes[node]['median_Y'])
            
            stats[community]['x_points'].append(x)
            stats[community]['y_points'].append(y)
            
            
    elif df is not None:
        for index, row in df.iterrows():
            community = row[column]
            if community not in stats.keys():
                stats[community] = {
                    'x_points': [],
                    'y_points': [],
                    'x_mean': 0.0,
                    'y_mean': 0.0,
                    'mean_distance_to_center': 0.0,
                    'standard_deviation': 0.0,
                }
                
            if 'median_X' not in row:
                continue
            
            x = float(row['median_X'])
            y = float(row['median_Y'])
            
            stats[community]['x_points'].append(x)
            stats[community]['y_points'].append(y)
    else:
        raise ValueError("You must provide a graph or a dataframe.")        
        
    for community in stats.keys():
        x_mean = np.mean(stats[community]['x_points'])
        y_mean = np.mean(stats[community]['y_points'])
        
        stats[community]['x_mean'] = x_mean
        stats[community]['y_mean'] = y_mean
        
        distances = [math.dist([x_mean, y_mean], [x, y]) for x, y in zip(stats[community]['x_points'], stats[community]['y_points'])]
        mean_distance_to_center = np.mean(distances)
        standart_deviation = np.std(distances)
        
        stats[community]['mean_distance_to_center'] = mean_distance_to_center
        stats[community]['standard_deviation'] = standart_deviation        
    
    return stats

def return_community_detection_duration(graph: nx.Graph, algorithm: str = 'louvain'):
    """Calcula o tempo de execução de um algoritmo de detecção de comunidades.

    Args:
        graph (nx.Graph): Grafo a ser analisado.
        algorithm (str, optional): Algoritmo de detecção de comunidades. Defaults to 'louvain'.

    Returns:
        float: Tempo de execução do algoritmo.
        
    Raises:
        ValueError: Se o algoritmo não for implementado.
    """
    start = time.time()
    
    if algorithm == 'louvain':
        partition = nx.community.louvain_communities(graph)
        modularity = nx.community.modularity(graph, partition)
    elif algorithm == 'greedy':
        partition = nx.community.greedy_modularity_communities(graph)
        modularity = nx.community.modularity(graph, partition)
    elif algorithm == 'edge_betweeness':
        sets = random.randint(2, len(graph))        
        partition = nx.community.edge_betweenness_partition(graph, sets)
        modularity = nx.community.modularity(graph, partition)
    elif algorithm == 'girvan_newman':
        comunidades = list(nx.community.girvan_newman(graph))
        partition = comunidades[0]
        modularity = nx.community.modularity(graph, partition)
        
        for i in range(1, len(comunidades)):
            partition_partial = comunidades[i]
            modularity_partial = nx.community.modularity(graph, partition_partial)
            
            if np.abs(modularity_partial) > np.abs(modularity):
                partition = partition_partial
                modularity = modularity_partial
    else:
        raise ValueError("Método não implementado.")
        
    end = time.time()
    
    return end - start, partition, modularity

def random_subgraph_by_porcentage(graph: nx.DiGraph, porcentage: float = 0.1, connected_only: bool = False) -> nx.DiGraph:
    """Gera um subgrafo aleatório de um grafo.

    Args:
        graph (nx.Graph): Grafo base.
        porcentage (float, optional): Porcentagem de subgrafo. Defaults to 0.1.

    Returns:
        nx.Graph: Subgrafo gerado.
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

def compute_and_export_the_data_from_subgraphs(graph, output_path: str = 'E://', output_file_name: str = 'subgraph_data.csv', columns: list = ['Nós', 'Links'], porcentage: float = 0.1):

    output_file = os.path.join(output_path, output_file_name)

    try:
        df_performance = pd.read_csv(output_file, encoding='utf-8')
    except:
        df_performance = pd.DataFrame(columns=columns)
        
    tempo_louvain, comunidades_louvain, modularidade_louvain = return_community_detection_duration(graph, 'louvain')
    tempo_greedy, comunidades_greedy, modularidade_greedy = return_community_detection_duration(graph, 'greedy')
    # tempo_edge_betweeness = return_community_detection_duration(graph, 'edge_betweeness')    
    tempo_girvan_newman, comunidades_girvan_newman, modularidade_girvan_newman = return_community_detection_duration(graph, 'girvan_newman')

    df_performance = df_performance._append({
        "Porcentagem de Arestas": porcentage,
        "Nós": graph.number_of_nodes(),
        "Links": graph.number_of_edges(),
        "Grau Médio": np.mean([graph.degree(node) for node in graph.nodes()]),
        "Densidade": nx.density(graph),
        "Cluster Global": nx.transitivity(graph),
        "Cluster Médio": nx.average_clustering(graph),
        "Tempo Louvain": tempo_louvain,
        "N° Comunidades Louvain": len(comunidades_louvain),
        "Modularidade Louvain": modularidade_louvain,
        "Tempo Greedy": tempo_greedy,
        "N° Comunidades Greedy": len(comunidades_greedy),
        "Modularidade Greedy": modularidade_greedy,
        # "Tempo Edge Betweeness": tempo_edge_betweeness,
        "Tempo Girvan-Newman": tempo_girvan_newman,
        "N° Comunidades Girvan-Newman": len(comunidades_girvan_newman),
        "Modularidade Girvan-Newman": modularidade_girvan_newman
    }, ignore_index=True)

    df_performance.to_csv(output_file, index=False, encoding='utf-8')  
    
itens = ["Porcentagem de Arestas", "Nós", "Links", "Grau Médio", "Densidade", "Cluster Global", "Cluster Médio", 
        "Tempo Louvain", "N° Comunidades Louvain", "Modularidade Louvain",
        "Tempo Greedy", "N° Comunidades Greedy", "Modularidade Greedy",
        "Tempo Girvan-Newman", "N° Comunidades Girvan-Newman", "Modularidade Girvan-Newman"]

porcentagens = [0.01, 0.025, 0.05, 0.1]

for item in porcentagens:
    grafo = random_subgraph_by_porcentage(grafo_twitter, item, connected_only=True)
    compute_and_export_the_data_from_subgraphs(grafo, columns=itens, porcentage=item, output_file_name="connected_subgraphs_data.csv")