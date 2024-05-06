k=10
cd ..
#Bert
python3 attack.py --device cuda --loss margin --dataset qnli --model textattack/bert-base-uncased-QNLI --k $k --pga 0 --n_positions 20 --select_pos_mode batch --size 1000

# #Albert
python3 attack.py --device cuda --loss margin --dataset qnli --model Alireza1044/albert-base-v2-qnli --k $k --pga 0 --n_positions 20 --select_pos_mode batch --size 1000

# #Roberta
python3 attack.py --device cuda --loss margin --dataset qnli --model textattack/roberta-base-QNLI --k $k --pga 0 --n_positions 20 --select_pos_mode batch --size 1000
