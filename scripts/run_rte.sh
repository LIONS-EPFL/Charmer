k=10
cd ..
#Bert
python3 attack.py --loss margin --device cpu --dataset rte --model textattack/bert-base-uncased-RTE --k $k --select_pos_mode batch --size 1000 --n_positions 20 --pga 0

# #Albert
python3 attack.py --loss margin --device cuda --dataset rte --model textattack/albert-base-v2-RTE --k $k --select_pos_mode batch --size 1000 --n_positions 20 --pga 0

# #Roberta
python3 attack.py --loss margin --device cuda --dataset rte --model textattack/roberta-base-RTE --k $k --select_pos_mode batch --size 1000 --n_positions 20 --pga 0