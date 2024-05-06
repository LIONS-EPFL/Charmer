import os
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type = str, required = True,
                        help = 'Directory where all the partial clusterers are stored...')
    parser.add_argument('--file_paths', nargs = '+',
                        help = 'Files that will be combined')
    return parser.parse_args()

def reconstruct_clusters(save_path, file_paths):
    num_clusters_added = 0
    clusters = {}
    word2cluster = {}
    cluster2representative = {}
    typo2cluster = {}
    word2freq = {}

    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            job_clusterer_dict = pickle.load(f)

        new_clusters = job_clusterer_dict['cluster']
        word2newcluster = job_clusterer_dict['word2cluster']
        newcluster2representative = job_clusterer_dict['cluster2representative']
        newtypo2cluster = job_clusterer_dict['typo2cluster']
        newword2freq = job_clusterer_dict['word2freq']

        for cluster_id in new_clusters:
            clusters[cluster_id + num_clusters_added] = new_clusters[cluster_id]
            cluster2representative[cluster_id + num_clusters_added] = newcluster2representative[cluster_id]

        for word in word2newcluster:
            word2cluster[word] = word2newcluster[word] + num_clusters_added
            word2freq[word] = newword2freq[word]

        for typo in newtypo2cluster:
            assert typo not in typo2cluster
            typo2cluster[typo] = newtypo2cluster[typo] + num_clusters_added

        num_clusters_added += len(new_clusters)

    save_dict = {'cluster': clusters, 'word2cluster': word2cluster, 
                            'cluster2representative': cluster2representative, 'typo2cluster': typo2cluster, 'word2freq': word2freq}
    for key in save_dict:
        if key not in ['typo2cluster', 'word2freq']:
            print(key)
            print(save_dict[key])

    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print("Saved!")

if __name__ == '__main__':
    args = parse_args()
    reconstruct_clusters(args.save_path, args.file_paths)


