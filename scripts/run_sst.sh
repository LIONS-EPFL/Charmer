k=10
cd ..
#Bert
python3 attack.py --device cuda --loss margin --dataset sst --model textattack/bert-base-uncased-SST-2 --k $k --n_positions 20 --select_pos_mode batch --size 1000 --pga 0
# #Albert
python3 attack.py --device cuda --loss margin --dataset sst --model textattack/albert-base-v2-SST-2 --k $k --n_positions 20 --select_pos_mode batch --size 1000 --pga 0
# #Roberta
python3 attack.py --device cuda --loss margin --dataset sst --model textattack/roberta-base-SST-2 --k $k --n_positions 20 --select_pos_mode batch --size 1000 --pga 0