import argparse
import pandas as pd
import tensorflow as tf
import torch
import tqdm
from dataloader import load_attack_dataset, get_class_num
from tqdm import tqdm
import utils
import os
import time
from copy import copy
import argparse
import pdb
import sys
from collections import OrderedDict
from baseline.llm.promptbench.config import ID_TO_LABEL 

'''
Disable all GPUS for tensorflow
Otherwise, the USE occupies all the memory
'''
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'
    
if __name__ == '__main__':
    utils.seed_everything(0)
    
    parser = argparse.ArgumentParser()
    ############### general argument
    general_args = parser.add_argument_group('general')
    parser.add_argument('--size', type = int, default = -1,help='number of sample, if -1, then use complete dataset')
    parser.add_argument('--device', type = str, default = 'cuda',help='cpu or cuda')
    parser.add_argument('--dataset', type = str, default = 'sst',help='name of the dataset to use')
    parser.add_argument('--model_name', type = str, default = 'textattack/bert-base-uncased-SST-2', help='name of the huggingface model to use')
    parser.add_argument('--debug', action = 'store_true', help='if debug, then it will output each step during the attack')
    parser.add_argument("--attack_name", type = str, default = 'charmer', help = 'charmer / textfooler/ ...')
    parser.add_argument("--resume", type = str, default = None, help = 'path of the file to continue feeding more samples')
    parser.add_argument("--sufix", type = str, default = '', help = 'sufix to the model folder')


    parser.add_argument('--llm_prompt_test', action = 'store_true', help='if true, then we test the accuracy of a particular prompt without doing attack')
    parser.add_argument('--llmselect', type = str, default = 'v1')
    
    ############### argument for our method
    charmer_args = parser.add_argument_group('charmer')
    parser.add_argument('--k', type = int, default = 1, help='max edit distance or max number of characters to be modified') 
    parser.add_argument('--lr', type = float, default = 0.2, help='learning rate')
    parser.add_argument('--decay', type = float, default = 0.95, help='decay rate of learning rate ')
    parser.add_argument('--n_positions', type = int, default = 5, help='number of positions to consider for the attack')
    parser.add_argument('--n_iter', type = int, default = 50, help='number of iterations for the attack') 
    parser.add_argument('--combine_layer', type = str, default = 'encoder', help='where to combine sentences with our variables')
    parser.add_argument('--loss', type = str, default = 'ce',help='losses implemented: "ce" (cross entropy) and "margin" (margin loss)')
    parser.add_argument('--tau', type = float, default = 5, help='the threshold in margin loss')
    parser.add_argument('--select_pos_mode', type = str, default = 'iterative', help="how to select the best positions to attack (iterative or batch)")
    parser.add_argument('--pga', type = int, default = 1, help="whether using pga into a simplex")
    parser.add_argument('--checker', type = str, default = None, help="whether using spell checker during the attack phase")
    parser.add_argument("--repeat_words", type = int, default = 1, help = 'modify the same word twice?')
    parser.add_argument("--min_word_length", type = int, default = 0, help = 'minimum word length')
    parser.add_argument("--modif_start", type = int, default = 1, help = 'modify start of the word')
    parser.add_argument("--modif_end", type = int, default = 1, help = 'modify end of word')
    parser.add_argument("--only_letters", type = int, default = 0, help = 'use only lowercase letters in the vocabulary?')

    args = parser.parse_args()
    args.device = torch.device(args.device) 
    
    '''
    Output folder and file definition
    '''
    if ('llama' in args.model_name) or ('flan-t5' in args.model_name) or ('vicuna' in args.model_name):
        args.llm = True
        folder_name  = os.path.join('results_attack','llm_classifier',args.dataset,args.model_name.split('/')[1])
        os.makedirs(folder_name, exist_ok=True)
        if args.attack_name == 'charmer':
            output_name = 'charmer'+f'_{args.k}_' + f'{args.n_iter}iter_' + args.combine_layer + '_' + args.loss + f'_pga{args.pga}' + f'_{args.size}' + f'{args.llmselect}llmselect'  + f'_npos{args.n_positions}' +'.csv'
            if args.tau==0:
                output_name=output_name.replace('margin','margintau0')
        elif args.attack_name in ['full_bruteforce','bruteforce', 'bruteforce_random']:
            output_name = args.attack_name + f'_k{args.k}.csv'
            if args.tau==0:
                output_name=output_name.replace('margin','margintau0')
        else:
            output_name = args.attack_name + '.csv'
    else:
        args.llm = False
        if args.checker is not None:
            folder_name  =os.path.join('results_attack','lm_classifier','basiclm_attack_checker',args.dataset)
        else:
            folder_name  =os.path.join('results_attack','lm_classifier','basiclm',args.dataset)
        os.makedirs(folder_name, exist_ok=True)
        if args.attack_name == 'charmer':
            output_name = args.model_name.split('/')[1] + f'_{args.k}_' + f'{args.n_iter}iter_' + args.combine_layer + '_' + args.loss + f'_pga{args.pga}' + '_' + args.select_pos_mode + f'{args.n_positions}_{args.size}'
        elif args.attack_name in ['bruteforce', 'bruteforce_random']:
            output_name = args.attack_name + f'_k{args.k}.csv'

        else:
            if 'textattack' in args.model_name:
                output_name = args.attack_name + '_' + args.dataset + '_' + args.model_name.split('/')[1].split('-')[0] 
            else:
                output_name = args.attack_name + '_' + args.dataset + '_' + args.model_name.split('/')[1]
            if args.attack_name == 'pruti':
                output_name += f'repeat{args.repeat_words}_k{args.k}' 
        output_name = output_name + args.sufix + '.csv'

    ## Load Dataset
    attack_dataset = load_attack_dataset(args.dataset)
    num_classes = get_class_num(args.dataset)
    model_wrapper = utils.load_model(args)
    if args.llm:
        label_map = {}
        for label in range(num_classes):
            label_str = ID_TO_LABEL['sst2' if args.dataset == 'sst' else args.dataset][label]
            label_map[label] = model_wrapper.tokenizer(label_str, return_tensors="pt", add_special_tokens = False, truncation = False).input_ids[0][0].item()
        print(label_map)
    else:
        if (args.model_name=='textattack/bert-base-uncased-MNLI' or args.model_name=='baseline/roben/model_output_agglomerative/MNLI_170605556888068') and args.dataset == 'mnli':  ## textattack label map for mnli
            label_map = {0: 1, 1: 2, 2: 0}
        elif (args.model_name=='textattack/roberta-base-MNLI') and args.dataset == 'mnli':  ## textattack label map for mnli
            label_map = {0: 2, 1: 1, 2: 0}
        else:
            label_map = {x:x for x in range(num_classes)}

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # get_attacker
    attacker = utils.get_attacker(model_wrapper,args)
        
    #load USE for cosine similarity
    with tf.device('/cpu:0'):
        use = utils.USE() 
        
    if args.only_letters:
        V = [-1] + [ord(c) for c in 'abcdefghijklmnopqrstuvwxyz']
    else:
        V = utils.get_vocabulary(attack_dataset, args.dataset)

    count,skip,succ,fail = 0,0,0,0
    if args.resume is not None:
        df = pd.read_csv(args.resume).to_dict(orient = 'list')
        start = len(df['original'])
    else:
        start = 0
        if args.checker is not None:
            df = {'original':[], 'perturbed':[], 'checker':[],'True':[],'Pred_original':[],'Pred_perturbed':[],'success':[], 'Dist_char':[], 'Dist_token':[], 'similarity':[], 'time':[]}
        else:
            df = {'original':[], 'perturbed':[],'True':[],'Pred_original':[],'Pred_perturbed':[],'success':[], 'Dist_char':[], 'Dist_token':[], 'similarity':[], 'time':[]}
        if args.llm:
            succ_hard = 0
            df['success_hard'] = []
    test_size = len(attack_dataset['label']) if args.size ==-1 else min(args.size,len(attack_dataset['label']))
    

    start_time_all = time.time()
    for idx in tqdm(range(start,test_size)):
        
        if args.dataset in ['agnews', 'rotten_tomatoes']:
            orig_S = attack_dataset['text'][idx]
            premise_S = None
            sentence = orig_S
            if args.checker is not None:
                checker_S = attacker.checker.correct_string(orig_S)
        elif args.dataset in  ['mnli', 'rte', 'qnli']:
            orig_S = attack_dataset['hypothesis'][idx]
            premise_S = attack_dataset['premise'][idx]
            sentence = (premise_S,orig_S)
            pred_label = torch.argmax(model_wrapper([sentence])[0]).item()
            if args.checker is not None:
                checker_S = (premise_S,attacker.checker.correct_string(orig_S))

        else:
            orig_S = attack_dataset['sentence'][idx]
            premise_S = None
            sentence = orig_S
            if args.checker is not None:
                checker_S = attacker.checker.correct_string(orig_S)
            
        orig_label = torch.tensor([label_map[attack_dataset['label'][idx]]]).to(args.device)
        if args.checker is not None:
            pred_label = torch.argmax(model_wrapper([checker_S])[0]).item()
        else:
            pred_label = torch.argmax(model_wrapper([sentence])[0]).item()
        
        df['original'].append(orig_S)
        df['True'].append(orig_label.item())
        df['Pred_original'].append(pred_label)
        
        '''
        We don't attack missclasified sentences
        '''
        if orig_label.item() != pred_label:
            print("skipping wrong samples....")
            skip += 1
            count += 1
            df['perturbed'].append(None)
            df['Pred_perturbed'].append(-1)
            df['success'].append(False)
            df['Dist_char'].append(-1)
            df['Dist_token'].append(-1)
            df['time'].append(-1)
            df['similarity'].append(-1)
            if args.llm:
                df['success_hard'].append(False)
            if args.checker is not None:
                df['checker'].append(None)

            continue

        if args.llm_prompt_test: # test the accuracy of a prompt
            continue
            
        # get targeted label for llm
        target_class  = None
        if args.llm:
            for _,value in label_map.items():
                if value!=orig_label:
                    target_class = value
                    
        start_time_single = time.time()
        if args.attack_name in ['charmer', 'bruteforce', 'bruteforce_random']:
            if args.checker is not None:
                adv_example,adv_label,checker_example= attacker.attack(V,orig_S,orig_label,premise_S,target_class)
            else:
                adv_example,adv_label= attacker.attack(V,orig_S,orig_label,premise_S,target_class)
        elif args.attack_name == 'full_bruteforce':
            adv_example,adv_label = attacker.attack_automaton(V,orig_S,orig_label,premise_S,target_class)
        else:
            if args.dataset in ['rte','qnli','mnli']:
                adv_example = attacker.attack(OrderedDict({'premise':premise_S, 'hypothesis': orig_S}), orig_label.item(),target_class).perturbed_result.attacked_text._text_input
                if 'hypothesis' in adv_example.keys():
                    adv_example = adv_example['hypothesis']
                else:
                    adv_example = orig_S
                adv_label = torch.argmax(model_wrapper([(sentence[0], adv_example)])[0]).item()

            elif args.dataset in ['sst', 'agnews']:
                adv_example = attacker.attack(orig_S, orig_label.item(),target_class).perturbed_result.attacked_text._text_input['text'] 
                adv_label = torch.argmax(model_wrapper([adv_example])[0]).item()
                
        end_time_single = time.time()
        
    
        df['Pred_perturbed'].append(adv_label)
        df['success'].append((adv_label!=orig_label).item())
        df['Dist_char'].append(utils.word_edit_distance(orig_S.lower(), adv_example.lower())[0])
        df['Dist_token'].append(utils.word_edit_distance(model_wrapper.tokenizer(adv_example)['input_ids'], model_wrapper.tokenizer(orig_S)['input_ids'])[0])
        df['time'].append(end_time_single - start_time_single)
        df['similarity'].append(use.compute_sim([orig_S], [adv_example])) 
        if args.checker is not None:
            df['checker'].append(checker_example)
            df['perturbed'].append(adv_example)
        else:
            df['perturbed'].append(adv_example)
        if adv_label != orig_label:
            succ+=1
            print('success', adv_example)     
        else:
            fail+=1
            print('fail',adv_example)
            
        if args.llm:
            if adv_label != orig_label and adv_label == target_class:
                df['success_hard'].append(True)
            else:
                df['success_hard'].append(False)
            print(f"[Succeeded /Failed / Skipped / OOM / Total] {succ} /{fail} / {skip} / {len(attack_dataset['label'])}")
        else:
            print(f"[Succeeded  /Failed / Skipped / OOM / Total] {succ} /{fail} / {skip} / {len(attack_dataset['label'])}")
        pd_df = pd.DataFrame(df)
        pd_df.to_csv(os.path.join(folder_name,output_name),index = False)
        print("time used: %s"%(end_time_single - start_time_single))
    end_time_all = time.time()
    print("Total time used: %s"%(end_time_all - start_time_all))
    
    if args.llm_prompt_test:
        print('Accuracy of this prompt: ', (test_size - skip)/test_size)