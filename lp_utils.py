import pandas as pd
import networkx as nx
import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm

"""
*************************************
*                                   *
*                                   *
*   UTILS FOR BENCHMARK EXECUTION   *
*                                   *
*                                   *
*************************************
"""

def disturbed_hin(G, split=0.1, random_state=None, edge_type=['event_date', 'event_event', 'event_location', 'event_person', 'event_org', 'event_theme'], type_feature='edge_type'):
    """
    G: hin;
    split: percentage to be cut from the hin;
    random_state: ;
    edge_type: listlike object of types of edges to be cut;
    type_feature: feature name of edge_type on your hin.
    """
    def keep_left(x, G):
        edge_split = x['type'].split('_')
        if G.nodes[x['node']]['node_type'] != edge_split[0]:
            x['node'], x['neighbor'] = x['neighbor'], x['node']
        return x
    # prepare data for type counting
    edges = list(G.edges)
    edge_types = [G[edge[0]][edge[1]][type_feature] for edge in edges]
    
    edges = pd.DataFrame(edges)
    edges = edges.rename(columns={0: 'node', 1: 'neighbor'})
    edges['type'] = edge_types
    edges = edges.apply(keep_left, G=G, axis=1)
    edges_group = edges.groupby(by=['type'], as_index=False).count().reset_index(drop=True)

    # preparar arestas para eliminar
    edges = edges.sample(frac=1, random_state=random_state).reset_index(drop=True)
    edges_group = edges_group.rename(columns={'node': 'count', 'neighbor': 'to_cut_count'})
    edges_group['to_cut_count'] = edges_group['to_cut_count'].apply(lambda x:round(x * split))
    to_cut = {}
    for index, row in edges_group.iterrows():
        if row['type'] in edge_type:
            to_cut[row['type']] = edges[edges['type'] == row['type']].reset_index(drop=True).loc[0:row['to_cut_count']-1]
                    
    G_disturbed = deepcopy(G)
    for key, tc_df in to_cut.items():
        for index, row in tc_df.iterrows():
            G_disturbed.remove_edge(row['node'],row['neighbor'])
    return G_disturbed, to_cut

def regularization(G, dim=512, embedding_feature: str = 'embedding', iterations=15, mi=0.85):
    nodes = []
    # inicializando vetor f para todos os nodes
    for node in G.nodes():
        G.nodes[node]['f'] = np.array([0.0]*dim)
        if embedding_feature in G.nodes[node]:
            G.nodes[node]['f'] = G.nodes[node][embedding_feature]*1.0
        nodes.append(node)
    pbar = tqdm(range(0, iterations))
    for iteration in pbar:
        random.shuffle(nodes)
        energy = 0.0
        # percorrendo cada node
        for node in nodes:
            f_new = np.array([0.0]*dim)
            f_old = np.array(G.nodes[node]['f'])*1.0
            sum_w = 0.0
            # percorrendo vizinhos do onde
            for neighbor in G.neighbors(node):
                w = 1.0
                if 'weight' in G[node][neighbor]:
                    w = G[node][neighbor]['weight']
                w /= np.sqrt(G.degree[neighbor])
                f_new = f_new + w*G.nodes[neighbor]['f']
                sum_w = sum_w + w
            if sum_w == 0.0: sum_w = 1.0
            f_new /= sum_w
            G.nodes[node]['f'] = f_new*1.0
            if embedding_feature in G.nodes[node]:
                G.nodes[node]['f'] = G.nodes[node][embedding_feature] * \
                    mi + G.nodes[node]['f']*(1.0-mi)
            energy = energy + np.linalg.norm(f_new-f_old)
        iteration = iteration + 1
        message = 'Iteration '+str(iteration)+' | Energy = '+str(energy)
        pbar.set_description(message)
    return G

def get_knn_data(G, node, embedding_feature: str = 'f'):
    knn_data, knn_nodes = [], []
    for node in nx.non_neighbors(G, node):
        if embedding_feature in G.nodes[node]:
            knn_data.append(G.nodes[node][embedding_feature])
            knn_nodes.append(node)
    return pd.DataFrame(knn_data), pd.DataFrame(knn_nodes)

from sklearn.neighbors import NearestNeighbors
def run_knn(k, G_restored, row, knn_data, knn_nodes, node_feature='node', embedding_feature='f'):
    if k == -1:
        k = knn_data.shape[0]
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(knn_data)
    indice = knn.kneighbors(G_restored.nodes[row[node_feature]][embedding_feature].reshape(-1, 512), return_distance=False)
    return [knn_nodes[0].iloc[indice[0][i]] for i in range(k)]

import multiprocess
def restore_hin(G, cutted_dict, n_jobs=-1, k=-1, node_feature='node', neighbor_feature='neighbor', node_type_feature='node_type', embedding_feature='f'):
    def process(start, end, G, key, value, return_dict, thread_id):
        value_thread = value.loc[start:(end-1)]
        restored_dict_thread = {'true': [], 'restored': [], 'edge_type': []}
        for index, row in tqdm(value_thread.iterrows(), total=value_thread.shape[0]):
            edge_to_add = key.split('_')
            edge_to_add[0] = row[node_feature]
            edge_to_add = [row[node_feature] if e == G.nodes[row[node_feature]][node_type_feature] and row[node_feature] != edge_to_add[0] else e for e in edge_to_add]
            knn_data, knn_nodes = get_knn_data(G, row[node_feature])
            knn_nodes['type'] = knn_nodes[0].apply(lambda x: G.nodes[x][node_type_feature])
            knn_data = knn_data[knn_nodes['type'].isin(edge_to_add)]
            knn_nodes = knn_nodes[knn_nodes['type'].isin(edge_to_add)]
            edge_to_add[1] = run_knn(k, G, row, knn_data, knn_nodes)
            restored_dict_thread['true'].append([row[node_feature], row[neighbor_feature]])
            restored_dict_thread['restored'].append(edge_to_add)
            restored_dict_thread['edge_type'].append(key)
        for key in restored_dict_thread.keys():
            _key = key + str(thread_id)
            return_dict[_key] = (restored_dict_thread[key])
    
    def split_processing(n_jobs, G, key, value, return_dict):
        split_size = round(len(value) / n_jobs)
        threads = []                                                                
        for i in range(n_jobs):                                                 
            # determine the indices of the list this thread will handle             
            start = i * split_size                                                  
            # special case on the last chunk to account for uneven splits           
            end = len(value) if i+1 == n_jobs else (i+1) * split_size                
            # create the thread
            threads.append(                                                         
                multiprocess.Process(target=process, args=(start, end, G, key, value, return_dict, i)))
            threads[-1].start() # start the thread we just created                  

        # wait for all threads to finish                                            
        for t in threads:
            t.join()

    if n_jobs == -1:
        n_jobs = multiprocess.cpu_count()
    restored_dict = {'true': [], 'restored': [], 'edge_type': []}
    return_dict = multiprocess.Manager().dict()

    for key, value in cutted_dict.items():
        split_processing(n_jobs, G, key, value, return_dict)
        return_dict = dict(return_dict)
        for thread_key in restored_dict.keys():
            for job in range(n_jobs):
                for res in return_dict[thread_key + str(job)]:
                    restored_dict[thread_key].append(res)
    return pd.DataFrame(restored_dict)