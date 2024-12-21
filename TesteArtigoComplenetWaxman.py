# %% [markdown]
# ## Importando as bibliotecas

# %%
import seaborn as sns; sns.set_theme()
import ControlledWaxmanGraph as cwg
import matplotlib.pyplot as plt
import AcessoryMethods
import networkx as nx
import pandas as pd
import numpy as np
import math
import time

from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
from scipy.optimize import linear_sum_assignment

# %%
# Definindo número de vértices do grafo
NUMBER_OF_VERTICES = 25000
# NUMBER_OF_VERTICES = int(input("Digite o número de vértices do grafo: "))

# Latitudes e longitudes máximas e mínimas
LAT_MIN = 36.9609718322753338
LAT_MAX = 42.1543159484863281
LON_MIN = -9.5470819473266033
LON_MAX = -6.1891422271728516

am = AcessoryMethods.AcessoryMethods()

# %%
ks = [40]
porcentage_edges_to_randomize = [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8, 16]
results_df = pd.DataFrame(columns=porcentage_edges_to_randomize, index=ks)

grafo : nx.Graph = nx.Graph()
grafo.add_nodes_from(range(NUMBER_OF_VERTICES))
radius = 1
angle_increment = 2 * np.pi / NUMBER_OF_VERTICES

for i, node in enumerate(grafo.nodes()):
    angle = i * angle_increment
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    grafo.nodes[node]['x'] = x
    grafo.nodes[node]['y'] = y
    grafo.nodes[node]['coords'] = (x, y)

for k in ks:        
    print(f"Gerando o grafo Waxman controlado com k={k}...")
    t0 = time.time()
    controle_grafo = cwg.ControlledWaxmanGraph(grafo, k, radius)
    print("Grafo gerado com sucesso em {0:.2f} minutos".format((time.time() - t0)/60))
    grafo_original = controle_grafo.graph.copy()
    results_df.index.name = f"edges randomized for k={k}"
    
    am.show_graph_metrics(grafo_original)

    # %%
    number_of_edges = grafo.number_of_edges()

    for index, porcentage in enumerate(porcentage_edges_to_randomize):
        print(f"Randomizando {porcentage}% das arestas...")
        if index == 0:
            edges = int(round((number_of_edges * porcentage / 100), 0))
        else:
            edges = int(round((number_of_edges * (porcentage - porcentage_edges_to_randomize[index-1]) / 100), 0))
        
        controle_grafo.randomize_edges(edges)
        grafo = controle_grafo.graph
        if index == 0:
            results_df.at[f"edges randomized for k={k}", porcentage] = edges
        else:
            results_df.at[f"edges randomized for k={k}", porcentage] = edges + results_df.at[f"edges randomized for k={k}", porcentage_edges_to_randomize[index-1]]
    
        # %%
        # geosocial = ga.GeoSocial(grafo, lat='y', lon='x')

        # grafo_geo = geosocial.return_geographic_graph_by_radius(distance, coords_str='coords')

        # %%
        print("Gerando as comunidades...")
        t0 = time.time()
        comunidade_social_waxman = nx.community.greedy_modularity_communities(grafo)
        comunidade_geo = nx.community.greedy_modularity_communities(grafo_original)
        print(f"Tempo para gerar as comunidades: {(time.time() - t0)/60} minutos")

        # %%
        print(f"Modularidade do grafo social: {nx.community.modularity(grafo, comunidade_social_waxman)}")

        # %%
        print(f"Modularidade do grafo geográfico: {nx.community.modularity(grafo_original, comunidade_geo)}")

        # %%
        node_community_map_social = am.map_community_nodes(comunidade_social_waxman)
        node_community_map_geo = am.map_community_nodes(comunidade_geo)

        am.print_progress_bar(0, grafo.number_of_nodes(), prefix = 'Progress:', suffix = 'Complete', length = 50)
        for i, node in enumerate(grafo.nodes()):
            grafo.nodes[node]['social_community'] = node_community_map_social[node]
            grafo.nodes[node]['geo_community'] = node_community_map_geo[node]
            am.print_progress_bar(i + 1, grafo.number_of_nodes(), prefix = 'Progress:', suffix = 'Complete', length = 50)
                        
        # %%
        am.plot_colored_communities(grafo, grafo_original, comunidade_social_waxman, comunidade_geo,
                                 with_labels=False, latitude='y', longitude='x',
                                 output_path='D:\\Documentos\\TCC\\Codigo_TCC\\ResultadosTesteWaxman\\',
                                 output_file_name=f'comunidades_{k}_porcentagem_{str(porcentage)}.png')

        # %%
        # nx.draw(grafo, pos=nx.spring_layout(grafo), node_color=[grafo.nodes[node]['social_community'] for node in grafo.nodes()], node_size=20, font_size=10, with_labels=False)

        # %%
        # nx.draw(grafo_geo, pos=nx.spring_layout(grafo_geo), node_color=[grafo.nodes[node]['social_community'] for node in grafo_geo.nodes()], node_size=20, font_size=10, with_labels=False)

        # %%
        jaccard_matrix = np.zeros((len(comunidade_social_waxman), len(comunidade_geo)))

        for i, com_social in enumerate(comunidade_social_waxman):
            for j, com_geo in enumerate(comunidade_geo):
                jaccard_matrix[i][j] = am.jaccard_similarity(set(com_social), set(com_geo))

        # %%
        jaccard_values = jaccard_matrix[jaccard_matrix !=0]
        upper_limit = np.max(jaccard_values)*1.2
        division = np.max(jaccard_values)/5

        plt.figure(figsize=(15, 6))
        plt.hist(jaccard_values, bins=100, color='lightgreen', edgecolor='black', log=True)
        plt.xlabel("Similaridade de Jaccard")
        plt.xticks(np.arange(0, upper_limit, division), rotation=90)
        plt.ylabel('Frequência (log)')
        plt.title('Distribuição da Similaridade de Jaccard')
        # plt.show()
        plt.close()

        # %%
        # Usando o algoritmo Hungarian para encontrar a melhor correspondência para a diagonal
        cost_matrix = -jaccard_matrix  # Inverter sinais para transformar em problema de maximização
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Reordenando as linhas e colunas para maximizar a diagonal
        permuted_matrix = jaccard_matrix[row_ind][:, col_ind]
        
        if jaccard_matrix.shape[0] > jaccard_matrix.shape[1]:
            missing_rows = [i for i in range(jaccard_matrix.shape[0]) if i not in row_ind]
            if missing_rows:
                additional_rows = jaccard_matrix[missing_rows, :]
                permuted_matrix = np.vstack([permuted_matrix, additional_rows])
        elif jaccard_matrix.shape[0] < jaccard_matrix.shape[1]:
            missing_columns = [i for i in range(jaccard_matrix.shape[1]) if i not in col_ind]
            if missing_columns:
                additional_columns = jaccard_matrix[:, missing_columns]
                permuted_matrix = np.hstack([permuted_matrix, additional_columns])
                
        

        # %%
        plt.figure(figsize=(24, 12))

        plt.subplot(1, 2, 1)
        sns.heatmap(jaccard_matrix, cmap='Oranges')
        plt.xlabel('Comunidades geográficas')
        plt.ylabel('Comunidades sociais')
        plt.title('Similaridade de Jaccard entre comunidades sociais e geográficas')

        plt.subplot(1, 2, 2)
        sns.heatmap(permuted_matrix, cmap='Oranges')
        plt.xlabel('Comunidades geográficas')
        plt.ylabel('Comunidades sociais')
        plt.title('Similaridade de Jaccard entre comunidades sociais e geográficas (permutada)')

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"D://Documentos//TCC//Codigo_TCC//ResultadosTesteWaxman//heatmap_{k}_porcentagem_{str(porcentage)}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # %%
        ami_jaccard = ami(am.assign_labels(comunidade_social_waxman, grafo.nodes()), am.assign_labels(comunidade_geo, grafo_original.nodes()))
        
        # Create a DataFrame to store the results

        # Save the result for the current k and porcentage
        results_df.at[k, porcentage] = ami_jaccard

        # Save the DataFrame to a CSV file
        results_df.to_csv(r'D:\Documentos\TCC\Codigo_TCC\ResultadosTesteWaxman\ami_jaccard_results.csv')


