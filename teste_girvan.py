import networkx as nx
import numpy as np
import random


def show_graph_metrics(graph):
    degrees = []

    for node in graph.nodes():
        degrees.append(nx.degree(graph, node))
        
    print(f"Nº de nós: {graph.number_of_nodes()}")
    print(f"Nº de links: {graph.number_of_edges()}")
    print(f"Grau médio: {np.mean(degrees)}")
    print(f"Densidade: {nx.density(graph)}")
    
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


if __name__ == "__main__":
    # Grafo de interações no Twitter
    grafo_twitter = nx.read_graphml("D:\\Documentos\\data_and_code\\all_data_lisbon\\grafo_twitter.graphml")
    show_graph_metrics(grafo_twitter)
    
    # Geração de subgrafo aleatório
    print("Gerando subgrafo aleatório...")
    subgrafo_twitter = random_subgraph_by_porcentage(grafo_twitter, porcentage=0.3, connected_only=True)
    show_graph_metrics(subgrafo_twitter)
    
    
    # Geração de comunidades girvan_newman
    print("Gerando comunidades girvan_newman...")
    communities = nx.algorithms.community.girvan_newman(subgrafo_twitter)
    
    # Exibição das comunidades
    for community in next(communities):
        print(community)
        print("\n")
