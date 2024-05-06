TASK_NAME=SST-2
CLUSTERER_PATH=clusterers/vocab100000_ed1.pkl

python3 agglom_clusters.py --gamma 0.3 --clusterer_path $CLUSTERER_PATH
# python3 agglom_clusters.py --gamma 0.3 --clusterer_path $CLUSTERER_PATH --job_id 0 --num_jobs 2 &
# python3 agglom_clusters.py --gamma 0.3 --clusterer_path $CLUSTERER_PATH --job_id 1 --num_jobs 2 && fg

python3 reconstruct_clusters.py --clusterer_dir $CLUSTERER_PATH
