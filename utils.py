from torch.nn.utils.rnn import pad_sequence
import torch
import os
import sys
import pdb
import time
import random
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, 'baseline/TextAttack/'))
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import LlamaForCausalLM, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForCausalLM

'''
Random
------------------------------------------------------------------------------------------------------------------
'''

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def print_vocabulary(V):
    for v in V:
        if v == -1:
            print('remove')
        else:
            print(v,chr(int(v)))

def get_vocabulary(dataset, dataset_name):
    '''
    get the characted volabulary from a dataset
    '''
    V = set([-1]) # Remove character operator
    if dataset_name in ['mnli', 'rte', 'qnli']:
        keyword = 'hypothesis'
    elif dataset_name in ['agnews', 'rotten_tomatoes']:
        keyword = 'text'
    else:
        keyword = 'sentence'
    for x in dataset[keyword]:
        V = V.union([ord(y) for y in set(x)])
    return list(V)


class USE:
    '''
    Universal Sentece Encoder

    Used for computing the similarity between the sentences before and after the attack
    '''
    def __init__(self):
        '''
        Just used for USE and NOTHING else
        '''
        import tensorflow as tf
        import tensorflow_hub as hub
        with tf.device('/cpu:0'):
            self.encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def compute_sim(self, clean_texts, adv_texts):
        import tensorflow as tf
        with tf.device('/cpu:0'):
            clean_embeddings = self.encoder(clean_texts)
            adv_embeddings = self.encoder(adv_texts)
            cosine_sim = tf.reduce_mean(tf.reduce_sum(clean_embeddings * adv_embeddings, axis=1))

            return float(cosine_sim.numpy())

def combined_emb_input(model_wrapper,U,S, labels = None):
    '''
    Combined output for the sentences in S with weights U
    '''
    E = []
    for i in range(len(S)):
        u = U[i]
        s = S[i]
        t = model_wrapper.tokenizer(s, padding = 'longest', return_tensors = 'pt')
        e = model_wrapper.model.embeddings(t['input_ids'].to(model_wrapper.device))
        E.append((e*(u.unsqueeze(-1))).sum(dim=0))
    if labels is not None:
        return model_wrapper.model(labels = labels, inputs_embeds = pad_sequence(E,batch_first=True))
    else:
        return model_wrapper.model(inputs_embeds = pad_sequence(E,batch_first=True))
    

'''
Edit distance related
------------------------------------------------------------------------------------------------------------------
'''

def all_strings_within_edit1(sequence, bases='ATCG'):
    """
    All edits that are one edit away from `sequence`
    using a dictionary of bases.

    Parameters
    ----------
    sequence: str
    bases: str

    Returns
    -------
    sequences: list of str

    """
    splits = [(sequence[:i], sequence[i:]) for i in range(len(sequence) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    # In the original code, transpose counts one edit distance
    # We count it as two edit distances, so it's not included here
    # transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in bases]
    inserts = [L + c + R for L, R in splits for c in bases]
    return deletes + replaces + inserts


def all_strings_within_editx(sequence, bases='ATCG', edit_distance=1):
    """
    Return all strings with a give edit distance away from

    Parameters
    ----------
    sequence: str
    bases: str
    edit_distance: int

    Returns
    -------
    sequences: set of str

    """
    if edit_distance == 0:
        return {sequence}
    elif edit_distance == 1:
        return set(all_strings_within_edit1(sequence, bases=bases))
    else:
        return set(
            e2 for e1 in all_strings_within_editx(
                sequence, bases=bases, edit_distance=edit_distance-1)
            for e2 in all_strings_within_edit1(e1, bases=bases)
        )
    

def all_strings_editx(sequence, bases='ATCG', edit_distance=1):
    """
    Return all strings of a give edit distance away from `sequence`

    Parameters
    ----------
    sequence: str
    bases: str
    edit_distance: int

    Returns
    -------
    result: generator of str

    """
    if edit_distance == 0:
        return [sequence]
    all_editx_minus1 = all_strings_within_editx(
        sequence, bases=bases, edit_distance=edit_distance-1)
    return (
        e2 for e1 in all_editx_minus1
        for e2 in all_strings_within_edit1(e1, bases=bases)
        if e2 not in all_editx_minus1
    )

def word_edit_distance(x, y):
    '''
    Compute the edit distance between x and y with dynamic programming

    returns the distance (edit_distance) and the distance matrix used to compute it.

    code from: https://stackoverflow.com/questions/66636450/how-to-implement-alignment-through-traceback-for-levenshtein-edit-distance
    '''
    rows = len(x) + 1
    cols = len(y) + 1
    distance = np.zeros((rows, cols), dtype=int)

    for i in range(1, rows):
        for k in range(1, cols):
            distance[i][0] = i
            distance[0][k] = k

    for col in range(1, cols):
        for row in range(1, rows):
            if x[row - 1] == y[col - 1]:
                cost = 0
            else:
                cost = 1
            distance[row][col] = min(distance[row - 1][col] + 1,
                                     distance[row][col - 1] + 1,
                                     distance[row - 1][col - 1] + cost)
    edit_distance = distance[-1][-1]
    return edit_distance, distance

def backtrace(first, second, matrix):
    '''
    Get the alignment (trace) from the distance matrix used in word_edit_distance
    code from: https://stackoverflow.com/questions/66636450/how-to-implement-alignment-through-traceback-for-levenshtein-edit-distance
    '''
    f = [char for char in first]
    s = [char for char in second]
    new_f, new_s = [], []
    row = len(f)
    col = len(s)
    trace = [[row, col]]

    while True:
        if f[row - 1] == s[col - 1]:
            cost = 0
        else:
            cost = 1

        r = matrix[row][col]
        a = matrix[row - 1][col]
        b = matrix[row - 1][col - 1]
        c = matrix[row][col - 1]

        if r == b + cost:
            # when diagonal backtrace substitution or no substitution
            trace.append([row - 1, col - 1])
            new_f = [f[row - 1]] + new_f
            new_s = [s[col - 1]] + new_s

            row, col = row - 1, col - 1

        else:
            # either deletion or insertion, find if minimum is up or left
            if r == a + 1:
                trace.append([row - 1, col])
                new_f = [f[row - 1]] + new_f
                new_s = [-1] + new_s

                row, col = row - 1, col

            elif r == c + 1:
                trace.append([row, col - 1])
                new_f = [-1] + new_f
                new_s = [s[col - 1]] + new_s

                row, col = row, col - 1

        # Exit the loop
        if row == 0 or col == 0:
            return trace, new_f, new_s
        
def mark_changes(first, second):
    '''
    given two lists first and second, align first and second filling the gaps with "-1" in both sentences (a and b)
    and marking where a change has been done with 1s in the vector p.
    example:
    first = ['it', "'s", 'a', 'charming', 'and', 'often', 'affecting', 'journey', '.'] 
    second = ['it', "'s", 'a', '%', 'harming', 'and', 'often', 'affecting', 'journey', '.']

    a = ['it', "'s", 'a', -1, 'charming', 'and', 'often', 'affecting', 'journey', '.']
    b = ['it', "'s", 'a', '%', 'harming', 'and', 'often', 'affecting', 'journey', '.']
    p = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    '''
    _,m = word_edit_distance(first,second)
    _, a,b = backtrace(first, second, m)
    p = []
    for i in range(len(a)):
        if a[i] == b[i]:
            p.append(0)
        else:
            p.append(1)

    return p, a, b

def perturbed_sentence_for_table(first, second):
    p, _, b = mark_changes(first, second)
    s = ''
    for i in range(len(b)):
        
        if not p[i]:
            s += b[i]
        elif p[i] and b[i] != -1 and (b[i] not in ['%', '$', '^']):
            s += '\\textcolor{red}{' + b[i] + '}'
        elif p[i] and b[i] != -1 and (b[i] not in ['_']):
            s += '\\textcolor{red}{\\' + b[i] + '}'
        else:
            pass
            #s += '\\textcolor{red}{\\_}'
    return s

        
def resize_embeddings(T, T2, E2, device,debug=False, agg = 'sum'):
    '''
    inputs:
    T: tokens of the un-preturbed sentence
    T2: Tokens of the perturbed sentence
    E2: embeddings of the perturbed sentence

    returns:
    E2_resized: resized embeddings to have the same sequence length as the original sentence.
    '''
    d, M = word_edit_distance(T, T2)
    trace, _, _ = backtrace(T, T2, M)
    prev=trace[-1][0]
    count=1
    E2_resized = torch.zeros((1,len(T), E2.shape[2]), device=device)
    counts = []
    id = 0
    if debug:
        print(len(T),E2.shape[1])
    for j,t in enumerate(reversed(trace[1:])):
        if debug:
            print(T[t[0]], T2[t[1]], t, id, prev)
        if prev == t[0]:
            if j!=0:
                count+=1
            E2_resized[:,id,:] += E2[:,t[1],:]
        else:
            prev = t[0]
            if agg == 'mean':
                E2_resized[:,id,:] /= count
            counts.append(count)
            count=1
            id+=1
            if agg=='max':
                E2_resized[:,id,:] = torch.max(E2[:,t[1],:], E2_resized[:,id,:])
            else:
                E2_resized[:,id,:] += E2[:,t[1],:]
    return E2_resized

'''
Attack related
------------------------------------------------------------------------------------------------------------------
'''

def generate_sentence(S,z,u, V,k=1, alternative = None):
    '''
    inputs:
    S: sentence that we want to modify
    z: location position
    u: selection character id
    V: vocabulary, list of UNICODE indices
    k: number of possible changes
    
    generate sentence with a single character modification at position z with character u
    '''
    spaces = ''.join(['_' for i in range(k)])
    xx = ''.join([spaces + s for s in S] + [spaces])
    new_sentence = [c for c in xx]
    mask = []
    for i in range(len(S)):
        mask += [0 for i in range(k)] + [1]
    mask+=[0 for i in range(k)]
    
    if type(z) == list:
        for p,c in zip(z,u):
            if V[c] != -1:
                new_sentence[p] = chr(V[c])
                mask[p] = 1
            else: 
                new_sentence[p] = '_'
                mask[p] = 0
    else:
        if V[u] != -1:
            if new_sentence[z] == chr(V[u]) and (alternative is not None) and alternative != -1:
                new_sentence[z] = chr(alternative)
                mask[z] = 1
            elif new_sentence[z] == chr(V[u]) and (alternative is not None) and alternative == -1:
                new_sentence[z] = '_'
                mask[z] = 0
            else:
                new_sentence[z] = chr(V[u])
                mask[z] = 1
        else: 
            new_sentence[z] = '_'
            mask[z] = 0
    
    new_sentence = [c if mask[i] else '' for i,c in enumerate(new_sentence)]
    new_sentence = ''.join(new_sentence)
    return new_sentence

def generate_all_sentences_at_z(S, z, V,k=1, alternative = -1):
    '''
    inputs:
    S: sentence that we want to modify
    z: location id
    V: vocabulary, list of UNICODE indices
    
    generates all the possible sentences by changing characters in the position z
    '''
    return [generate_sentence(S,z,u, V,k, alternative=alternative) for u in range(len(V))]

def generate_all_sentences(S,V,subset_z = None,k=1, alternative = None):
    '''
    inputs:
    S: sentence that we want to modify
    V: vocabulary, list of UNICODE indices
    subset_z: subset of positions to consider
    k: number of character modifications
    alternative: in the case len(V)=1, character to consider for switchings when the character to change is
    the one in the volcabulary
    
    generates all the possible sentences by changing characters
    '''
    out = []
    if subset_z is None:
        subset_z = range((k+1)*len(S) + k)
    for z in subset_z:
        out += generate_all_sentences_at_z(S, z, V, k, alternative=alternative)
    return out



'''
LLM related
------------------------------------------------------------------------------------------------------------------
'''

class llm_input_processor(object):
    def __init__(self,dataset,premise,model_name = 'llama'):
        self.dataset = dataset
        self.premise = premise

        if 'llama' in model_name or 'flan-t5-large' in model_name:
            if dataset == 'sst2' or dataset == 'sst': 
                self.promptbegin="Is the given review positive or negative? "
                self.promptend='The answer is'  
    
            elif dataset == 'qnli':
                self.promptbegin="Does the sentence answer the question? Answer with yes or no." \
                    + ' Question: '\
                    +   self.premise \
                    + ' Sentence: '
                self.promptend=' The answer is'  

                self.insert_index = 2 # the sentence will be inserted after this position  of the self.promptlist
            
            elif dataset == 'rte':
                self.promptbegin=  self.premise + " Based on the paragraph above can we conclude the following sentence, answer with yes or no."
                self.promptend=' The answer is'  
            else: 
                NotImplementedError

        elif 'vicuna' in model_name:
            if dataset == 'sst2' or dataset == 'sst': 
                self.promptbegin="Analyze the tone of this statement and respond with either positive or negative: "
                self.promptend='The answer is:' 
            elif dataset == 'rte':
                self.promptbegin=  self.premise + " Based on the paragraph above can we conclude the following sentence, answer with yes or no."
                self.promptend=' The answer is'  
            elif dataset == 'qnli':    
                self.promptbegin= self.premise + " Based on the question above, does the following sentence answer the question? "
                self.promptend=' Answer with yes or no. The answer is' 
            else:
                NotImplementedError
                
        else:
            NotImplementedError

    def addprompt(self,S):
        return  self.promptbegin + S + self.promptend

def concat_prompt_question_label(T_promptbegin,T_questions,T_promptend,T_label):     
    '''
    For llm, concat tokens of  (promptbegin,questions,promptend,label)
    '''
    T_all= {}
    
    T_all['input_ids'] = torch.cat([torch.cat([
        T_promptbegin['input_ids'],
        i.unsqueeze(0),
        T_promptend['input_ids'],
        T_label['input_ids'],
        ],dim=-1) for i in T_questions['input_ids']],dim=0) #shape(number_of_pos,sequenceLength)
    
    T_all['attention_mask'] = torch.cat([torch.cat(
        [T_promptbegin['attention_mask'],
            i.unsqueeze(0),
            T_promptend['attention_mask'],
            T_label['attention_mask'],
            ],dim=-1) for i in T_questions['attention_mask']],dim=0) #shape(number_of_pos,sequenceLength)
    return T_all
         
         
   
def get_llm_embeddings(model, input_ids):
    if isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, T5ForConditionalGeneration):
        return model.encoder.embed_tokens(input_ids)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

   
def get_llm_encoder(model, input_ids,layer_idx):
    if isinstance(model, LlamaForCausalLM):
        inputs_embeds = model.model.embed_tokens(input_ids)
        attention_mask = torch.ones_like(input_ids)
        batch_size, seq_length = input_ids.shape
        past_key_values_length = 0
        attention_mask = model.model._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        hidden_states = inputs_embeds
        for _, decoder_layer in enumerate(model.model.layers[:layer_idx+1]):
            layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_value=None,
            output_attentions=model.config.output_attentions,
            use_cache=False,
        )
            hidden_states = layer_outputs[0]
        return hidden_states,attention_mask
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def margin_loss_lm(logits, true_class):
    '''
    Standard margin loss for classification
    '''
    #maximum different than true class
    max_other,_ = (torch.cat((logits[:,:true_class], logits[:,true_class+1:]), dim=-1)).max(dim=-1)
    return max_other - logits[:,true_class]

class margin_loss_lm_batched():
    def __init__(self,reduction = 'None'):
        self.reduction = reduction
    
    def __call__(self,logits, true_classes):
        '''
        Standard margin loss for classification
        '''
        L = torch.cat([margin_loss_lm(l.unsqueeze(0), t) for l,t in zip(logits,true_classes)], dim=0)
        if self.reduction == 'mean':
            return torch.mean(L)
        elif self.reduction == 'sum':
            return torch.sum(L)
        else:
            return L


class CrossEntropyLoss_negative():
    def __init__(self):
        pass
    def __call__(self,logits, true_classes):
        return  -1 * torch.nn.CrossEntropyLoss(reduce='mean')(logits,true_classes)


class margin_loss_llm():
    def __init__(self,target_class,tau):
        self.target_class = target_class
        self.tau = tau
    def __call__(self,logits, true_classes):
        if self.tau==0:
            return -1 * (logits[:, true_classes]  - logits[:, self.target_class])
        else:
            return -1 * (logits[:, true_classes]  - logits[:, self.target_class] + self.tau).clamp(min=0) 

def load_model(args):
    '''load model'''
    print("loading model: ",  args.model_name)
    
    if args.llm:
        from utils_llm_inference import Inference
        model_wrapper = Inference(args)
    else:
        import textattack
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name,ignore_mismatched_sizes=True)
        model = model.eval() #IMPORTANT eval
        model = model.to(args.device)
        if args.attack_name == 'charmer':
            model = rename_model(model)
        model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        
    return model_wrapper

def rename_model(model):
    '''
    IMPORTANT:
    for our code to work every model needs to have the attributes embeddigns, encoder and classifier, we wrap this appropriately in the next lines
    '''
    if type(model).__name__ == 'AlbertForSequenceClassification':
        model.encoder = lambda **kwargs: model.albert(**kwargs).pooler_output
        model.classifier_wrapped = model.classifier
        model.embeddings = model.albert.embeddings.word_embeddings
    elif type(model).__name__ == 'BertForSequenceClassification':
        model.encoder = lambda **kwargs: model.bert(**kwargs).pooler_output
        model.classifier_wrapped = model.classifier
        model.embeddings = model.bert.embeddings.word_embeddings
    elif type(model).__name__ == 'RobertaForSequenceClassification':
        model.encoder = lambda **kwargs: model.roberta(**kwargs)[0][:,0,:]
        model.classifier_wrapped = lambda x: model.classifier(x.unsqueeze(1))
        model.embeddings = model.roberta.embeddings.word_embeddings
    elif type(model).__name__ == 'XLNetForSequenceClassification':
        model.embeddings = model.transformer.word_embedding
        model.encoder = lambda x: model.sequence_summary(model.transformer(x).last_hidden_state)
        model.classifier_wrapped = model.logits_proj
    elif type(model).__name__ == 'LlamaForCausalLM':
        pass
    else:
        raise NotImplementedError 
    
    return model

def get_attacker(model_wrapper,args):
    if args.attack_name in ['charmer', 'full_bruteforce', 'bruteforce', 'bruteforce_random']:
        from charmer import Charmer
        attack = Charmer(model_wrapper,args)
    else:
        import textattack
        if args.attack_name == 'textfooler':
            attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper,llm=args.llm)
        elif args.attack_name == 'textbugger':
            attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper,llm=args.llm)
        elif args.attack_name == 'bertattack':
            attack = textattack.attack_recipes.BERTAttackLi2020.build(model_wrapper,llm=args.llm)
        elif args.attack_name == 'deepwordbug':
            attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper,llm=args.llm)
        elif args.attack_name == 'baer':
            attack = textattack.attack_recipes.BAEGarg2019.build(model_wrapper,llm=args.llm) 
        elif args.attack_name == 'pruti':
            attack = textattack.attack_recipes.Pruthi2019.build(model_wrapper,max_num_word_swaps=args.k, repeat = bool(args.repeat_words)) 
        else:
            raise NotImplementedError
    return attack