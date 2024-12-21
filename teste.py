import networkx as nx

# Definindo número de vértices do grafo
NUMBER_OF_VERTICES = 1000

k = 10
grafo_watts = nx.random_graphs.watts_strogatz_graph(NUMBER_OF_VERTICES, k, 0.1)
