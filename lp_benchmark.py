import pickle5 as pickle
import time

from lp_utils import disturbed_hin
from lp_utils import regularization
from lp_utils import restore_hin

path = '/Users/paulocarmo/repositories/nyt-lp/'

def execution(G, algorithm, split, iteration, edge_types):
    G_disturbed, cutted = disturbed_hin(G, split=split, random_state=(1 + iteration), edge_type=edge_types)
    
    if algorithm == 'regularization':
        start_time = time.time()
        G_disturbed = regularization(G_disturbed, iterations=15, mi=0.85)
        restored_df = restore_hin(G_disturbed, cutted)
        with open("{}results/execution_time.txt".format(path), 'a') as f:
            f.write(f'{algorithm},{split},{iteration},{edge_types},{(time.time() - start_time)}\n')
        restored_df.to_csv("results/knn_results_{}_{}_{}_{}.csv".format(algorithm, split, edge_types, iteration), index=False)

if __name__ == '__main__':
    network_name = "hin_v0"
    splits = [0.6]
    edge_types = ['_id_where']
    algorithms = ['regularization']

    with open("{}.gpickle".format(network_name), "rb") as fh:
        G = pickle.load(fh)

    # regularization
    for split in splits:
        for iteration in range(1):
            for algorithm in algorithms: 
                execution(G, algorithm, split, iteration, edge_types)