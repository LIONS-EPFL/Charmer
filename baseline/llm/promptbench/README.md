python main.py --model google/flan-t5-large --dataset mmlu --attack semantic --shot 0 --generate_len 20


python main.py --model google/flan-t5-large --dataset sst2 --attack deepwordbug --shot 0 --generate_len 20 --debug

python main.py --model google/flan-t5-large --dataset bool_logic --attack textbugger --shot 0 --generate_len 20 --debug

without debug, this will get INFO:__main__:Original acc: 53.00%, attacked acc: 37.00%, dropped acc: 16.00%




python main.py --model google/flan-t5-large --dataset qnli --attack deepwordbug --shot 0 --generate_len 20 --debug