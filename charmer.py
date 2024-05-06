import torch
import pdb
import numpy as np
from scipy.special import binom
from context import ctx_noparamgrad
from types import SimpleNamespace  
import utils
import random
import sys
import os
import copy

dir_path = os.path.dirname(os.path.realpath(__file__))

class Charmer:
    """character-level attack generates adversarial examples on text.
    Args:
        model_wrapper, includes infomarmation of model and tokenizer
        tokenizer
        args: arguments received in the main function
    """
    def __init__( self,model_wrapper,args):
        self.model_wrapper = model_wrapper
        self.args = args
        self.device = args.device
        if self.args.loss == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none') 
        elif self.args.loss == 'ce_neg':
            self.criterion = utils.CrossEntropyLoss_negative()
        elif not args.llm and self.args.loss == 'margin':
            self.criterion = utils.margin_loss_lm_batched(reduction='none')
        else:
            pass
            #raise NotImplementedError
        if self.args.checker is not None:
            if self.args.checker == 'ScRNN':
                sys.path.insert(0, os.path.join(dir_path, 'baseline/roben'))
                from run_glue import get_recoverer
                if self.args.dataset == 'sst':
                    task_name = 'SST-2'
                elif self.args.dataset == 'rte':
                    task_name = 'RTE'
                elif self.args.dataset == 'mnli':
                    task_name = 'MNLI'
                elif self.args.dataset == 'qnli':
                    task_name = 'QNLI'
                    
                args = SimpleNamespace(**{'output_dir':'.', 'recoverer':'scrnn','tc_dir':'baseline/roben/tc_data', 'task_name':task_name})
                self.checker = get_recoverer(args)
                self.checker.correct_string = lambda x: self.checker.recover(x)
            elif self.args.checker == 'roben-cc':
                sys.path.insert(0, os.path.join(dir_path, 'baseline/roben'))
                from run_glue import get_recoverer
                args = SimpleNamespace(**{'output_dir':'.', 'recoverer':'clust-rep','clusterer_path' : 'baseline/roben/clusterers/vocab100000_ed1.pkl', 'lm_num_possibilities' : 10})
                self.checker = get_recoverer(args)
                self.checker.correct_string = lambda x: self.checker.recover(x)
            elif self.args.checker == 'roben-agglom':
                sys.path.insert(0, os.path.join(dir_path, 'baseline/roben'))
                from run_glue import get_recoverer
                args = SimpleNamespace(**{'output_dir':'.', 'recoverer':'clust-rep','clusterer_path' : 'baseline/roben/clusterers/vocab100000_ed1_gamma0.3.pkl', 'lm_num_possibilities' : 10})
                self.checker = get_recoverer(args)
                self.checker.correct_string = lambda x: self.checker.recover(x)
            else:
                raise NotImplementedError
        
    def attack(self,V,orig_S,orig_label,premise = None,target_class=None, return_us = False):
        """ Iteratively attack a maximum of k times and return the adversarial sentence. 
        Firstly, we select the top locations.
        Secondly, we optimizer the u variable.

        Inputs:
            orig_S: str, the original sentence
            orig_label: TorchTensor, the original sentence
        Returns:
            (The attacked sentence, the attacked label)
        """
        self.premise = premise
        self.V = V
        
        if self.args.llm:
            self.T_label = {'input_ids':torch.tensor([[orig_label]]).to(self.device), 
                            'attention_mask': torch.tensor([[1]]).to(self.device)}
            self.llm_input_processor = utils.llm_input_processor(self.args.dataset,premise,self.args.model_name)
            self.T_promptbegin =  self.model_wrapper.tokenizer(self.llm_input_processor.promptbegin,
                                        return_tensors = 'pt',
                                        add_special_tokens = True,
                                        truncation = True).to(self.device)
            
            self.T_promptend =  self.model_wrapper.tokenizer(self.llm_input_processor.promptend,
                                        return_tensors = 'pt',
                                        add_special_tokens = False).to(self.device)
            if self.args.loss == 'margin':
                self.criterion = utils.margin_loss_llm(torch.tensor([target_class]).to(self.device),self.args.tau)
        constraints_loc = self.update_constrained_positions(orig_S,orig_S,[1 for _ in range(2*len(orig_S) + 1)])
        if self.args.k < 1:
            if self.args.checker is not None:
                return orig_S, orig_label.item(), self.checker.correct_string(orig_S)
            else:
                return orig_S, orig_label.item()
        with ctx_noparamgrad(self.model_wrapper.model):        
            for _ in range(self.args.k):
                if sum(constraints_loc)==0:
                    if self.args.checker is not None:
                        SS = [self.checker.correct_string(orig_S)]
                    else:
                        SS = [orig_S]
                    id_best = 0
                    u = [1]
                    label_best = orig_label.item()
                    break
                S_old = orig_S
                if self.args.attack_name == 'charmer':
                    if self.args.n_positions == -1:
                        print(len(orig_S))
                        #if the length is too large, it goes out of memory, we are going to restrict it to 120 positions
                        n=120
                        if len(orig_S)<=n:
                            subset_z = None
                        else:
                            subset_z = np.random.choice(len(orig_S),size = n,replace=False).tolist()
                    else:
                        subset_z = self.get_top_n_locations(orig_S,orig_label,constraints_loc = constraints_loc)
                    if self.args.pga:
                        if self.args.checker is not None:
                            SS, SS_,u,l = self.attack_ours_pga(orig_S, orig_label,subset_z = subset_z)
                        else:
                            SS,u,l = self.attack_ours_pga(orig_S, orig_label,subset_z = subset_z)
                    else:
                        #Bruteforce in the positions selected
                        if self.args.checker is not None:
                            SS_ = utils.generate_all_sentences(orig_S,self.V,subset_z,1)
                            SS = [self.checker.correct_string(s) for s in SS_]
                            _,u,l = self.attack_brute_force(orig_S, orig_label,SS=SS)
                        else:
                            SS,u,l = self.attack_brute_force(orig_S, orig_label,subset_z = subset_z)

                elif self.args.attack_name == 'bruteforce':
                    SS,u,l = self.attack_brute_force(orig_S, orig_label,bs=1024)
                
                elif self.args.attack_name == 'bruteforce_random':
                    _,SS,u,l = self.attack_brute_force_random(orig_S, orig_label,bs=1024)
                if (self.args.attack_name in ['bruteforce', 'bruteforce_random']) or not self.args.pga:
                    idxs = list(torch.topk(u.flatten(),1).indices)
                else:
                    idxs = list(torch.topk(u.flatten(),5).indices)
                if self.args.llm:
                    if self.args.dataset in ['mnli', 'rte', 'qnli']:
                        new_sentences = [(self.premise,SS[j]) for j in idxs]
                    else:
                        new_sentences = [SS[j] for j in idxs]
                    adv_score = self.model_wrapper(new_sentences)
                    adv_score = torch.stack([i for i in adv_score]).to(self.device)
                    loss = self.criterion(adv_score, self.T_label['input_ids'][0])
                else:
                    new_sentences = [SS[j] for j in idxs]
                    if self.args.dataset in ['mnli', 'rte', 'qnli']:
                        T = self.model_wrapper.tokenizer([self.premise for j in range(len(idxs))],new_sentences,return_tensors = 'pt', padding='longest', add_special_tokens = True, truncation = True)
                    else:
                        T = self.model_wrapper.tokenizer(new_sentences,return_tensors = 'pt', padding='longest', add_special_tokens = True, truncation = True)
                    if 'token_type_ids' in T.keys():
                        adv_score = self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device), token_type_ids = T['token_type_ids'].to(self.device)).logits
                    else:
                        adv_score = self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device)).logits
                    loss = self.criterion(adv_score,orig_label.repeat(adv_score.shape[0]))
                adv_label = torch.argmax(adv_score, dim=-1)
                id_best_batch = torch.argmax(loss, dim=0) #from 0 to 4
                id_best = idxs[id_best_batch] #from 0 to len(SS)
                label_best = adv_label[id_best_batch].item()
                if self.args.checker is not None:
                    orig_S = SS_[id_best]
                else:
                    orig_S = SS[id_best]
                constraints_loc = self.update_constrained_positions(S_old,orig_S,constraints_loc)
                if label_best!=orig_label or sum(constraints_loc)==0:
                    break
                
        if self.args.checker is not None:
            return orig_S, label_best, SS[id_best]
        else:
            if return_us:
                return orig_S, label_best, u, SS
            else:
                return orig_S, label_best
            
    def attack_automaton(self,V,orig_S,orig_label,premise = None,target_class=None, return_us = False):
        '''
        Try every possibility within levenshtein distance k
        '''
        self.premise = premise
        self.V = V
        V_aux = [] #all_strings_within_editx already considers deletions and needs characters, not codes for characters
        for s in V:
            if s!=-1:
                V_aux.append(chr(s))
        with ctx_noparamgrad(self.model_wrapper.model):        
            for k in range(1,self.args.k+1):
                SS = list(utils.all_strings_within_editx(orig_S,V_aux,k))
                print(len(SS), binom(k*(len(orig_S) + 1) + len(orig_S),k)*(len(V)**k))
                _,u,l = self.attack_brute_force(orig_S, orig_label,SS = SS)
                idx = np.argmax(u.cpu().numpy())
                adv_example = SS[idx]
                if self.args.dataset in ['mnli', 'rte', 'qnli']:
                    out = self.model_wrapper((premise,adv_example))
                else:
                    out = self.model_wrapper(adv_example)
                adv_label = torch.argmax(out, dim=-1).item()
                if adv_label!=orig_label:
                    break
        return adv_example, adv_label


    
    def align_tokens_with_sentence(self,S,tokens, normalize = True, tokenized = True):
        '''
        given a sentence S and a list of tokens obtained from that semtence, maps 
        every character in the token to a position in the sentence
        '''
        alignment = []
        if normalize:
            normalMap = {'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A',
                'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'ª': 'A',
                'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
                'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
                'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
                'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
                'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O',
                'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'º': 'O',
                'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U',
                'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
                'Ñ': 'N', 'ñ': 'n',
                'Ç': 'C', 'ç': 'c',
                '§': 'S',  '³': '3', '²': '2', '¹': '1'}
            normalize = str.maketrans(normalMap)
            S = S.translate(normalize) 
        i=0
        while len(tokens):
            a = []
            t = tokens[0]
            if tokenized:
                if t[:2] == '##':#Deal with sufix tokens
                    t = t[2:]
                elif t[0] == 'Ġ':
                    t = t[1:]
                elif t[0] == '▁':
                    t = t[1:]
            for j in range(len(t)):
                while S[i] == ' ':
                    i+=1
                if t[j] == S[i]:
                    a.append(i)
                    i+=1
                else:
                    print('not a match')
                    tokens = tokens[1:]
                    alignment.append(a)
                    i+= len(t) - j 
                    break
            tokens = tokens[1:]
            alignment.append(a)
        return alignment

    def get_locations_from_tokens(self,S,T,best):
        '''
        given the tokens T from sentence S and the ids in "best".
        get the positions of every character of every token in the "best" subset
        '''
        tokens = self.model_wrapper.tokenizer.convert_ids_to_tokens(T['input_ids'][0].tolist())[1:-1]
        alignment = self.align_tokens_with_sentence(S,tokens)
        l = []
        for i in best:
            prev = 2*(alignment[i][0])
            post = 2*(alignment[i][-1] + 1)
            if prev not in l:
                l.append(prev)
            if post not in l:
                l.append(post)
            for j in alignment[i]:
                l.append(2*j + 1)
        return l
    
    def update_constrained_positions(self,S_old,S_new,constraint_old):
        '''
        returns a vector with 2*len(S_new) + 1 boolean values indicating whether or not 
        the position can be modified in the next modification of S_new

        for generating the starting constraints, just feed S_old=S_new and constraint_old a vector of all ones
        
        input:
        S_old = old sentence
        S_new = modified sentence
        constraint_old = old vector of constraints
        repeat_words = modify a word more than once?
        min_word_length = minimum length of a word to be modified
        modif_start =  modify first character of a word
        modif_end =  modify last character of a word

        example:
        S_old = "Hello my friend"
        S_new = "Helloo my friend"
        (meaning of each position on top for reference)
                        [_,H,_,e,_,l,_,l,_,o,_, ,_,m,_,y,_, ,_,f,_,r,_,i,_,e,_,n,_,d,_]
        constraint_old = [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1]
        default parameters
                [_,H,_,e,_,l,_,l,_,o,_,o,_, ,_,m,_,y,_, ,_,f,_,r,_,i,_,e,_,n,_,d,_]
        output = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1]
        '''
        result = [1 for _ in range(2*len(S_new) + 1)]

        words_old = S_old.split()
        words_new = S_new.split()

        '''
        Cases:
        len(S_new) == len(S_old): modified character
        len(S_new) = len(S_old) - 1: removed character
        len(S_new) = len(S_old) + 1: inserted a character
        '''
        modified_position = -1
        if len(S_new) == len(S_old):
            for i,c in enumerate(S_new):
                if c != S_old[i]:
                    modified_position = 2*i + 1
        elif len(S_new) == (len(S_old) - 1):
            for i,c in enumerate(S_new):
                if c != S_old[i]:
                    modified_position = 2*i + 1
                    break
            if modified_position == -1:
                modified_position = 2*len(S_old) - 1
        else:
            for i,c in enumerate(S_old):
                if c != S_new[i]:
                    modified_position = 2*i
                    break
            if modified_position == -1:
                modified_position = 2*len(S_old)
        assert len(S_new) in [len(S_old), len(S_old)-1, len(S_old)+1]
        
        if modified_position != -1:
            if not self.args.repeat_words:
                pos_words = self.align_tokens_with_sentence(S_old,words_old, normalize=False, tokenized=False)
                for p in pos_words:
                    if modified_position>=2*p[0] and modified_position<=2*p[-1] + 2:
                        for i in range(2*p[0],2*p[-1] + 3):
                            if i>=0 and i<len(constraint_old):
                                constraint_old[i] = 0
                    

            if modified_position%2 == 0: #insertion
                for i in range(modified_position):
                    result[i] = constraint_old[i]
                if not self.args.repeat_words:
                    result[modified_position]=0
                    result[modified_position+1]=0
                for i in range(modified_position,len(constraint_old)):
                    result[i+2] = constraint_old[i]
            else:
                if len(S_old) == len(S_new):
                    for i in range(len(constraint_old)):
                        if i!= modified_position:
                            result[i] = constraint_old[i]
                        else:
                            result[i] = 0
                else:
                    for i in range(modified_position-1):
                        result[i] = constraint_old[i]
                    for i in range(modified_position-1,len(result)):
                        result[i] = constraint_old[i+2]
        else:
            result = constraint_old

        
        for i in range(len(S_new)):
            if S_new[i] == ' ':
                if (i == len(S_new)-1 or S_new[i+1] == ' ') and (self.args.min_word_length > 0 or not self.args.modif_start):
                    result[2*i+2] = 0

        short_words = [0 if len(w) >= self.args.min_word_length else 1 for w in words_new]
        pos_words = self.align_tokens_with_sentence(S_new,words_new, normalize=False, tokenized=False)
        for i,p in enumerate(pos_words):
            if short_words[i]:
                for j in range(2*p[0]-1,2*p[-1] + 4):
                    if j>=0 and j<len(result):
                        result[j] = 0
            if not self.args.modif_start:
                result[2*p[0]+1] = 0
                result[2*p[0]] = 0
                if p[0]>0:
                    result[2*p[0]-1] = 0
            if not self.args.modif_end:
                result[2*p[-1]+1] = 0
                result[2*p[-1]+2] = 0
                if 2*p[0]+3<len(result):
                    result[2*p[0]+3] = 0

        return result

    def get_llmloss_from_questions(self,criterion,questions = None,T_questions = None):
        """
        the input can be eiter the raw questions or the token of question
        """
        if T_questions == None:
            T_questions =  self.model_wrapper.tokenizer(questions, 
                                    padding = 'longest',
                                    return_tensors = 'pt', 
                                    add_special_tokens = False, 
                                    truncation = True).to(self.device)
        
        T_all = utils.concat_prompt_question_label(self.T_promptbegin,
                                                    T_questions,
                                                    self.T_promptend,
                                                    self.T_label)
        
        pred=  self.model_wrapper.token_to_logit(T_all['input_ids'],T_all['attention_mask'])      
        losses = []
        
        for i, tokens in enumerate(T_all['input_ids']):
            if pred.shape[1]==1:
                loss_slice = slice(0,1) # when the model ouput only a logit in a single position, i.e., T5
            else:
                loss_slice = slice(pred.shape[1] -  len(self.T_label['input_ids'][0])  - 1, pred.shape[1] - 1)
                # loss_slice = slice(pred.shape[1]-2,pred.shape[1]-1,1)
            loss = criterion(pred[i,loss_slice,:], self.T_label['input_ids'][0]).mean(dim=-1)
            losses.append(loss)
        losses = torch.FloatTensor(losses)

        if not self.args.debug:
            del T_all,T_questions,pred #release cuda memory
            with self.device:
                torch.cuda.empty_cache()
                            
        return losses
        
    def get_top_n_locations(self,S,label,bs = 1024, constraints_loc = None):
        '''
        Get the locations where the loss is changed the most after introducing an space character.
        '''                

        if self.args.llm:
            if self.args.select_pos_mode =='iterative': raise NotImplementedError
            elif self.args.select_pos_mode =='batch_jailbreak':
                if constraints_loc is not None:
                    subset_z_aux = []
                    for i in range(len(constraints_loc)):
                        if constraints_loc[i]:
                            subset_z_aux.append(i)
                    SS = utils.generate_all_sentences(S,[ord(' ')], subset_z_aux, 1, alternative=-1)
                else:
                    SS = utils.generate_all_sentences(S,[ord(' ')], None,1, alternative=-1)
                T = self.model_wrapper.tokenizer([self.model_wrapper.tokenizer.apply_chat_template([{'role':'user','content': s},
                                                                  {'role':'assistant', 'content': label}],tokenize=False) for s in SS], add_special_tokens = True, padding = 'longest',return_tensors = 'pt',padding_side = 'left', truncation = True)
                if 'token_type_ids' in T.keys():
                    pred = self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device),token_type_ids = T['token_type_ids'].to(self.device)).logits
                else:
                    pred = self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device)).logits
                loss = self.get_llmloss_2(pred,T)
                idxs = list(torch.topk(loss,min(self.args.n_positions,len(loss))).indices)
                if constraints_loc is not None:
                    return [subset_z_aux[i] for i in idxs]
                else:
                    return idxs
            with torch.no_grad():
                self.orig_T =  self.model_wrapper.tokenizer(S, 
                                        return_tensors = 'pt', 
                                        add_special_tokens = False, 
                                        truncation = True).to(self.device)
                
                #v5 #mask in token level, then generate in character level, two forward pass!
                if self.args.llmselect == 'v5' or self.args.llmselect == 'v51':
                    tokens_num = len(self.orig_T['input_ids'][0])
                    T_questions = {}
                    T_questions['input_ids'] = self.orig_T['input_ids'].repeat(tokens_num, 1)
                    T_questions['attention_mask'] = self.orig_T['attention_mask'].repeat(tokens_num,1)
                    T_questions['input_ids'].fill_diagonal_(self.model_wrapper.tokenizer.unk_token_id)
                    losses = self.get_llmloss_from_questions(criterion= self.criterion, questions = None, T_questions = T_questions)
                    if self.args.llmselect == 'v5':
                        number_top_token = 5
                        number_positions = 20
                    elif self.args.llmselect == 'v51': 
                        number_top_token = 10 
                        number_positions = 40
                    else:
                        raise NotImplementedError
                    position_list = []
                    for top_tokens_indice in list(torch.topk(losses, min (number_top_token,losses.shape[0])).indices): 
                        top_tokens = self.model_wrapper.tokenizer.decode(self.orig_T['input_ids'][0][top_tokens_indice])
                        start = S.find(top_tokens) 
                        end = start + len(top_tokens)
                        position_list += list(range(start*2, end*2,1))
                        if len(position_list)>=number_positions:
                            break
                    questions = utils.generate_all_sentences(S,[ord(' ')], position_list,1)
                    losses = self.get_llmloss_from_questions(criterion =self.criterion,questions = questions)
                    return [position_list[i.item()] for i in torch.topk(losses,self.args.n_positions).indices]
                          
                #v4 #mask in token level
                if self.args.llmselect == 'v4':
                    tokens_num = len(self.orig_T['input_ids'][0])
                    T_questions = {}
                    T_questions['input_ids'] = self.orig_T['input_ids'].repeat(tokens_num, 1)
                    T_questions['attention_mask'] = self.orig_T['attention_mask'].repeat(tokens_num,1)
                    T_questions['input_ids'].fill_diagonal_(self.model_wrapper.tokenizer.unk_token_id)
                    losses = self.get_llmloss_from_questions(criterion= self.criterion, questions = None, T_questions = T_questions)
                    position_list = []
                    for top_tokens_indice in list(torch.topk(losses,2).indices):
                        top_tokens = self.model_wrapper.tokenizer.decode(self.orig_T['input_ids'][0][top_tokens_indice])
                        start = S.find(top_tokens) 
                        end = start + len(top_tokens)
                        temp = list(range(start*2, end*2,1))
                        if len(temp)> 5:
                            position_list+=random.sample(temp,5)
                        else:
                            position_list+=temp      
                    return position_list

                #v3 # mask in sentence level
                if self.args.llmselect == 'v3':
                    questions = []
                    indexs = []
                    start_search_idx = 0
                    for i in S.split(' '):
                        if len(i) == 0: 
                            continue
                        begin = S.find(i,start_search_idx,len(S))
                        end = begin + len(i)
                        q = S[:begin] + S[end:]
                        questions.append(q)
                        indexs.append((begin,end))
                        start_search_idx=end
                    losses = self.get_llmloss_from_questions(criterion= self.criterion, questions = questions)
                    position_list = []
                    for each in torch.topk(losses,2).indices[:1]:
                        temp = list(range(indexs[each][0]*2, indexs[each][1]*2,1))
                        if len(temp)> 5:
                            position_list+=random.sample(temp,5)
                        else:
                            position_list+=temp      
                        
                    return position_list

                #v2 sample half
                elif self.args.llmselect == 'v2' or self.args.llmselect == 'v21' or self.args.llmselect == 'v22':
                    if self.args.llmselect == 'v2':
                        subset_z = list(range(2*len(S)+1))
                        random.shuffle(subset_z)
                        subset_z =subset_z[:len(S)]
                    elif self.args.llmselect == 'v21':
                        subset_z = list(range(2*len(S)+1))
                        random.shuffle(subset_z)
                        subset_z =subset_z[: (2*len(S)+1)//3]
                    elif self.args.llmselect == 'v22':
                        randomlist = list(np.random.randint(low = 1,high=3,size=len(S)))
                        subset_z = []
                        for i in range(len(S)):
                            if randomlist[i]//2 == 0 :
                                subset_z.append(2*i)
                            else:
                                subset_z.append(2*i+1)
                    else:
                        NotImplementedError

                    questions = utils.generate_all_sentences(S,[ord(' ')], subset_z,1)
                    losses = self.get_llmloss_from_questions(criterion= self.criterion, questions = questions)
                    return [subset_z[i.item()] for i in torch.topk(losses,self.args.n_positions).indices]
                
                #v1 original version, sample all
                elif self.args.llmselect == 'v1':
                    subset_z  =None
                    questions = utils.generate_all_sentences(S,[ord(' ')], subset_z,1)
                    losses = self.get_llmloss_from_questions(criterion= self.criterion, questions = questions) #TODO, this should be changed to get_llmloss in jailbreaking
                    return list(torch.topk(losses,self.args.n_positions).indices)
                else:
                    NotImplementedError
        else:
            if self.args.select_pos_mode=='iterative':
                with torch.no_grad():
                    if constraints_loc is not None:
                        subset_z_aux = []
                        for i in range(len(constraints_loc)):
                            if constraints_loc[i]:
                                subset_z_aux.append(i)
                        SS = utils.generate_all_sentences(S,[ord(' ')], subset_z_aux, 1, alternative=-1)
                    else:
                        SS = utils.generate_all_sentences(S,[ord(' ')], None,1, alternative=-1)
                    '''if mode is iterative, each perturbed sentence is fed in batches of bs to the model if not, all of the perturbed sentences are fed in a big batch, this can be memory expensive for lengthy sentences '''
                    pred = []
                    for i in range(len(SS)//bs + 1):
                        if self.premise is not None:
                            T = self.model_wrapper.tokenizer([self.premise for j in range(i*bs,min((i+1)*bs, len(SS)))], SS[i*bs:min((i+1)*bs, len(SS))], padding = 'longest',return_tensors = 'pt', add_special_tokens = True, truncation = True)
                        else:
                            T = self.model_wrapper.tokenizer(SS[i*bs:min((i+1)*bs, len(SS))], padding = 'longest',return_tensors = 'pt', add_special_tokens = True, truncation = True)
                        if 'token_type_ids' in T.keys():
                            pred.append(self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device),token_type_ids = T['token_type_ids'].to(self.device)).logits)
                        else:
                            pred.append(self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device)).logits)
                    pred = torch.cat(pred, dim=0)
                    loss = self.criterion(pred,label.repeat(pred.shape[0]))
                    idxs = list(torch.topk(loss,min(self.args.n_positions,len(loss))).indices)
                    if constraints_loc is not None:
                        return [subset_z_aux[i] for i in idxs]
                    else:
                        return idxs
            elif self.args.select_pos_mode=='batch':
                with torch.no_grad():
                    if constraints_loc is not None:
                        subset_z_aux = []
                        for i in range(len(constraints_loc)):
                            if constraints_loc[i]:
                                subset_z_aux.append(i)
                        SS = utils.generate_all_sentences(S,[ord(' ')], subset_z_aux, 1, alternative=-1)
                    else:
                        SS = utils.generate_all_sentences(S,[ord(' ')], None,1, alternative=-1)
                    if self.premise is not None:
                        T = self.model_wrapper.tokenizer([self.premise for j in range(len(SS))],SS, padding = 'longest',return_tensors = 'pt', add_special_tokens = True, truncation = True)
                    else:
                        T = self.model_wrapper.tokenizer(SS, padding = 'longest',return_tensors = 'pt', add_special_tokens = True, truncation = True)
                    if 'token_type_ids' in T.keys():
                        pred = self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device),token_type_ids = T['token_type_ids'].to(self.device)).logits
                    else:
                        pred = self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device)).logits
                    loss = self.criterion(pred,label.repeat(pred.shape[0]))
                    idxs = list(torch.topk(loss,min(self.args.n_positions,len(loss))).indices)
                    if constraints_loc is not None:
                        return [subset_z_aux[i] for i in idxs]
                    else:
                        return idxs

            if self.args.select_pos_mode=='hierarchical':
                #Not working yet
                with torch.no_grad():
                    number_top_token = 10 
                    if self.premise is not None:
                        T_ = self.model_wrapper.tokenizer(S,return_tensors = 'pt', add_special_tokens = True, truncation = True)
                        T = self.model_wrapper.tokenizer(self.premise,S,return_tensors = 'pt', add_special_tokens = True, truncation = True)
                    else:
                        T_ = self.model_wrapper.tokenizer(S,return_tensors = 'pt', add_special_tokens = True, truncation = True)
                        T = self.model_wrapper.tokenizer(S,return_tensors = 'pt', add_special_tokens = True, truncation = True)
                    losses = []
                    for i in range(T_['input_ids'].shape[1]-2):
                        TT = copy.deepcopy(T)
                        TT['input_ids'][0,1+i] = self.model_wrapper.tokenizer.unk_token_id
                        if 'token_type_ids' in TT.keys():
                            pred = self.model_wrapper.model(input_ids = TT['input_ids'].to(self.device), attention_mask = TT['attention_mask'].to(self.device),token_type_ids = TT['token_type_ids'].to(self.device)).logits
                        else:
                            pred = self.model_wrapper.model(input_ids = TT['input_ids'].to(self.device), attention_mask = TT['attention_mask'].to(self.device)).logits
                        losses.append(self.criterion(pred,label).item())
                    print(losses)
                    print(self.model_wrapper.tokenizer.convert_ids_to_tokens(T_['input_ids'][0].tolist())[1:-1])
                    token_idxs = list(torch.topk(torch.tensor(losses),min(number_top_token,len(losses))).indices)
                    subset_z = self.get_locations_from_tokens(S,T_,token_idxs)
                    print(S)
                    print(subset_z, 2*len(S)+1)
                    SS = utils.generate_all_sentences(S,[ord(' ')], subset_z,1)
                    if self.premise is not None:
                        T = self.model_wrapper.tokenizer([self.premise for j in range(len(SS))],SS, padding = 'longest',return_tensors = 'pt', add_special_tokens = True, truncation = True)
                    else:
                        T = self.model_wrapper.tokenizer(SS, padding = 'longest',return_tensors = 'pt', add_special_tokens = True, truncation = True)
                    if 'token_type_ids' in T.keys():
                        pred = self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device),token_type_ids = T['token_type_ids'].to(self.device)).logits
                    else:
                        pred = self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device)).logits
                    loss = self.criterion(pred,label.repeat(pred.shape[0]))
                    return list(torch.topk(loss,min(len(loss),self.args.n_positions)).indices)
            elif self.args.select_pos_mode=='random':
                return torch.randint(0,2*len(S)+1,(self.args.n_positions,)).tolist()      
    
    def simplex_projection(self, u):
        u_ = sorted(u, reverse=True) 
        ck = [0 for i in range(len(u_))]
        for i in range(len(u_)):
            if i==0:
                ck[i] = u_[i]-1
            else:
                ck[i] = (ck[i-1]*i + u_[i])/(i+1)

        k = len(u)
        while ck[k-1]>= u_[k-1] and k>0:
            k-=1
        tau = ck[k-1]

        return [max(x-tau, 0) for x in u]

    
    def attack_ours_pga(self, S, label, subset_z = None, init = None):
        '''
        inputs:
        clsf: classfier from hugging face
        S: sentence that we want to modify
        combine_layer: the layer we combine different sentences, choices: 'encoder' or 'emb'
        loss: "ce" for cross entropy or "margin" for margin loss
        
        do gradient descent on the u vector with signed gradient descent but scaling to maintain the sum1 restriction
        use barrier functions to maintain the [0,1] interval restriction

        combine_layer == emb2: Use the weird combination method at the embedding level using the alignment coming from the token-level edit distance.
        
        llm: bool, whether use large language model, choices: True or False
        llmloss: str,  the type of loss, choices: 'ce', 'margin'
        '''
        if subset_z is None:
            subset_z = range(2*len(S) + 1)

        if 'encoder' in self.args.combine_layer:
            if self.args.checker is not None:
                encoder_output, SS, SS_ = self.perturbed_all(S,subset_z = subset_z)
            else:
                encoder_output, SS = self.perturbed_all(S,subset_z = subset_z)

            encoder_output = encoder_output.detach()
        elif self.args.combine_layer in ['emb', 'emb2']:
            if self.args.checker is not None:
                emb_output, SS, SS_ = self.perturbed_all(S,subset_z = subset_z)
            else:
                emb_output, SS = self.perturbed_all(S,subset_z = subset_z)
            emb_output = emb_output.detach()
        else:
            raise NotImplementedError
        
        if init is None:
            u = torch.ones([len(SS),1]).to(self.device)/len(SS)
        else:
            u = init.to(self.device)
        
        l = []
        lr = self.args.lr
        
        for i in range(self.args.n_iter):
            u.requires_grad = True
            
            if self.args.llm:                  
                if self.args.combine_layer in ['emb', 'emb2']:
                    combined_emb = (u.unsqueeze(-1).to(torch.float16)*emb_output).sum(dim=0, keepdim=True)
                    out = self.model_wrapper.emb_to_logit(inputs_embeds = combined_emb)
                elif 'encoder' in self.args.combine_layer:
                    # encoder_output
                    combined = (u.unsqueeze(-1).to(torch.float16)*encoder_output).sum(dim=0, keepdim=True)
                    hidden_states = combined
                    if self.layer_idx+1<len(self.model_wrapper.model.model.layers):
                        for _, decoder_layer in enumerate(self.model_wrapper.model.model.layers[self.layer_idx+1:]):
                            layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=self.attention_mask_encode[:1],
                            position_ids=None,
                            past_key_value=None,
                            output_attentions=self.model_wrapper.model.config.output_attentions,
                            use_cache=False,
                        )
                            hidden_states = layer_outputs[0]
                    hidden_states = self.model_wrapper.model.model.norm(hidden_states)
                    hidden_states = hidden_states
                    out = self.model_wrapper.model.lm_head(hidden_states)#[:,-1,:]

                else:
                    raise NotImplementedError
                loss = self.criterion(out[0,-1:],self.T_label['input_ids'][0])
            else:
                if self.args.combine_layer == 'encoder':
                    combined = (u*encoder_output).sum(dim=0, keepdim=True)
                    out = self.model_wrapper.model.classifier_wrapped(combined)
                elif self.args.combine_layer in ['emb', 'emb2']:
                    combined_emb = (u.unsqueeze(-1)*emb_output).sum(dim=0, keepdim=True)
                    out = self.model_wrapper.model(inputs_embeds = combined_emb).logits
                else:
                    raise NotImplementedError
                loss = self.criterion(out,label)
            loss.backward(retain_graph = True)
            l.append(loss.item())

            g = u.grad.clone() 
            u = u.detach()
            
            u_hat = u.clone() + lr*g
            u = torch.tensor(self.simplex_projection(u_hat.flatten().tolist()), device = g.device).view(g.shape)

            lr*=self.args.decay
            
            if self.args.debug:
                if not self.args.llm:
                    print(i, loss.item(), torch.topk(u.flatten(),5).values)
                else:
                    if i %10 !=0:
                        continue
        if self.args.checker is not None:
            return SS, SS_ ,u,l
        else:
            return SS,u,l
        
    
    def attack_brute_force(self, S, label,bs = 1024, subset_z = None, SS = None):
        '''
        inputs:
        clsf: classfier from hugging face
        S: sentence that we want to modify
        combine_layer: the layer we combine different sentences, choices: 'encoder' or 'emb'
        loss: "ce" for cross entropy or "margin" for margin loss
        
        do gradient descent on the u vector with signed gradient descent but scaling to maintain the sum1 restriction
        use barrier functions to maintain the [0,1] interval restriction

        combine_layer == emb2: Use the weird combination method at the embedding level using the alignment coming from the token-level edit distance.
        
        llm: bool, whether use large language model, choices: True or False
        llmloss: str,  the type of loss, choices: 'ce', 'margin'
        '''
    
        with torch.no_grad():
            if SS is None:
                SS = utils.generate_all_sentences(S,self.V,subset_z,1)
            if bs==-1:
                bs = len(SS)

            if self.args.llm:
                loss = []
                for i in range(len(SS)//bs+ (len(SS)%bs>0)):
                    loss.append(self.get_llmloss_from_questions(criterion= self.criterion, questions = SS[i*bs: min((i+1)*bs, len(SS))]))
                loss = torch.cat(loss, dim=0)
                u = torch.zeros(len(SS))
                u[torch.argmax(loss)] = 1
            else:
                pred = []
                #print(len(SS),bs,len(SS)//bs)
                for i in range(len(SS)//bs + 1):
                    #print(i*bs,min((i+1)*bs, len(SS)))
                    if self.premise is not None:
                        T = self.model_wrapper.tokenizer([self.premise for j in range(i*bs,min((i+1)*bs, len(SS)))], SS[i*bs:min((i+1)*bs, len(SS))], padding = 'longest',return_tensors = 'pt', add_special_tokens = True, truncation = True)
                    else:
                        T = self.model_wrapper.tokenizer(SS[i*bs:min((i+1)*bs, len(SS))], padding = 'longest',return_tensors = 'pt', add_special_tokens = True, truncation = True)
                    if 'token_type_ids' in T.keys():
                        pred.append(self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device),token_type_ids = T['token_type_ids'].to(self.device)).logits)
                    else:
                        pred.append(self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device)).logits)
                pred = torch.cat(pred, dim=0)
                loss = self.criterion(pred,label.repeat(pred.shape[0]))
                u = torch.zeros(len(SS))
                u[torch.argmax(loss)] = 1
        return SS,u, None
    
    def attack_brute_force_random(self, S, label,bs = 1024):
        with torch.no_grad():
            SS = utils.generate_all_sentences(S,self.V,None,1)
            if bs==-1:
                bs = len(SS)
            #print(len(SS))
            B = np.random.choice(SS, min(bs, len(SS)), replace = False).tolist()
            if self.premise is not None:
                T = self.model_wrapper.tokenizer([self.premise for j in range(len(B))], B, padding = 'longest',return_tensors = 'pt', add_special_tokens = True, truncation = True)
            else:
                T = self.model_wrapper.tokenizer(B, padding = 'longest',return_tensors = 'pt', add_special_tokens = True, truncation = True)
            if 'token_type_ids' in T.keys():
                pred = self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device),token_type_ids = T['token_type_ids'].to(self.device)).logits
            else:
                pred = self.model_wrapper.model(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device)).logits
            loss = self.criterion(pred,label.repeat(pred.shape[0]))
            u = torch.zeros(len(B))
            u[torch.argmax(loss)] = 1
        return SS, B,u, None

    def perturbed_all(self,S, subset_z = None,k=1,alternative = None):

        '''
        inputs:
        clsf: classfier from hugging face
        S: sentence that we want to modify
        V: vocabulary, list of UNICODE indices
        '''
        
        if subset_z is None:
            subset_z = range(2*len(S) + 1)
        if self.args.checker is not None:
            SS_ = utils.generate_all_sentences(S,self.V,subset_z,k,alternative=alternative)
            SS = [self.checker.correct_string(s) for s in SS_]
        else:
            SS = utils.generate_all_sentences(S,self.V,subset_z,k,alternative=alternative)


        
        with torch.no_grad():
            if self.args.llm:
                T_questions =  self.model_wrapper.tokenizer(SS, 
                                        padding = 'longest',
                                        return_tensors = 'pt', 
                                        add_special_tokens = False, 
                                        truncation = True).to(self.device)
                
                T_all = utils.concat_prompt_question_label(self.T_promptbegin,
                                                           T_questions,
                                                           self.T_promptend,
                                                           self.T_label)
                
                if 'encoder' in self.args.combine_layer:
                    self.layer_idx = int(self.args.combine_layer[-1])
                    E,attention_mask_encode = utils.get_llm_encoder(self.model_wrapper.model,(T_all['input_ids']),
                                                                    self.layer_idx)
                    self.attention_mask_encode = attention_mask_encode
                    return E, SS 
                elif self.args.combine_layer == 'emb':  
                    E = utils.get_llm_embeddings(self.model_wrapper.model,(T_all['input_ids']))
                    return E, SS
                else:
                    raise NotImplementedError    
            else:
                if self.args.combine_layer == 'encoder':
                    if self.premise is not None:
                        T= self.model_wrapper.tokenizer([self.premise for i in range(len(SS))], SS, padding = 'longest', return_tensors = 'pt', add_special_tokens = True, truncation = True)
                    else:
                        T= self.model_wrapper.tokenizer(SS, padding = 'longest', return_tensors = 'pt', add_special_tokens = True, truncation = True)
                    if 'token_type_ids' in T.keys():
                        encoder_output = self.model_wrapper.model.encoder(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device),token_type_ids = T['token_type_ids'].to(self.device))
                    else:
                        encoder_output = self.model_wrapper.model.encoder(input_ids = T['input_ids'].to(self.device), attention_mask = T['attention_mask'].to(self.device))
                    if self.args.checker is not None:
                        return encoder_output, SS, SS_
                    else:
                        return encoder_output, SS
                elif self.args.combine_layer == 'emb':  
                    if self.premise is not None:
                        T= self.model_wrapper.tokenizer([self.premise for i in range(len(SS))], SS, padding = 'longest', return_tensors = 'pt')
                    else:
                        T= self.model_wrapper.tokenizer(SS, padding = 'longest', return_tensors = 'pt')
                    E = self.model_wrapper.model.embeddings(T['input_ids'].to(self.device))
                    if self.args.checker is not None:
                        return E, SS, SS_
                    else:
                        return E, SS
                
                elif self.args.combine_layer == 'emb2':  
                    if self.premise is not None:
                        T= self.model_wrapper.tokenizer([self.premise for i in range(len(SS))], SS, add_special_tokens = True, truncation = True)
                        T_0 = self.model_wrapper.tokenizer(self.premise, S, add_special_tokens = True, truncation = True)
                    else:
                        T= self.model_wrapper.tokenizer(SS, add_special_tokens = True, truncation = True)
                        T_0 = self.model_wrapper.tokenizer(S, add_special_tokens = True, truncation = True)
                    E = torch.cat([utils.resize_embeddings(T_0['input_ids'], t, self.model_wrapper.model.embeddings(torch.tensor([t]).to(self.device)),self.device, debug=False) for t in T['input_ids']], dim=0)
                    if self.args.checker is not None:
                        return E, SS, SS_
                    else:
                        return E, SS
                else:
                    raise NotImplementedError