k=20
cd ..
#Bert
python3 attack.py --device cuda --loss margin --dataset agnews --model textattack/bert-base-uncased-ag-news --k $k --pga 0 --n_positions 20 --select_pos_mode iterative --size 1000

#Albert
python3 attack.py --device cuda --loss margin --dataset agnews --model textattack/albert-base-v2-ag-news --k $k --pga 0 --n_positions 20 --select_pos_mode iterative --size 1000 

#Roberta
python3 attack.py --device cuda --loss margin --dataset agnews --model textattack/roberta-base-ag-news --k $k --pga 0 --n_positions 20 --select_pos_mode iterative --size 1000 