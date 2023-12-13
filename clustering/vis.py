import matplotlib.pyplot as plt
import pickle
from shapely.geometry import Polygon
import sys

if __name__ == "__main__":
    dataset       = sys.argv[1] 
    cluster_num   = sys.argv[2]
    train_seq_num = sys.argv[3]
    algorithm     = sys.argv[4]
    
    if algorithm == "agglomerative":
        path = "./" + dataset + "/clustering/" + algorithm + "_" + train_seq_num + "_100_" + cluster_num + "_None"
    else:
        path = "./" + dataset + "/clustering/" + algorithm + "_" + train_seq_num + "_100_" + cluster_num
    print(path)
    cluster = pickle.load(open(path, "rb"))
    fig, ax = plt.subplots()
    for polygon in cluster:
        patch = plt.Polygon(polygon, edgecolor='b', facecolor='none')
        ax.add_patch(patch)
    ax.set_xlim(0, int(train_seq_num))
    ax.set_ylim(0, int(train_seq_num))
    plt.savefig(dataset + "_" + train_seq_num + "_" + cluster_num + "_" + algorithm + '.png', dpi=300, bbox_inches='tight')