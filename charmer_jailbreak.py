import torch
import numpy as np
from context import ctx_noparamgrad
import time 
from charmer import Charmer
import logging
import gc
import utils
import sys
import pdb
import random
from utils_jailbreak import get_logits, target_loss, SuffixManager

class Charmer_jail(Charmer):
    """character-level attack generates adversarial examples on text.
    Args:
        model_wrapper, includes infomarmation of model and tokenizer
        tokenizer
        args: arguments received in the main function
    """
    def __init__( self,model_wrapper,V,conv_template,args):
        super(Charmer_jail, self).__init__(model_wrapper,args)
        self.reg = 0
        self.V = V
        self.conv_template=conv_template

    def generate(self,model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = self.args.generate_len
            
        input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
        # logging.info('!!heredebug'+tokenizer.decode(input_ids[0])[-10:])
        attn_masks = torch.ones_like(input_ids).to(model.device)
        output_ids = model.generate(input_ids, 
                                    attention_mask=attn_masks, 
                                    generation_config=gen_config,
                                    pad_token_id=tokenizer.pad_token_id)[0]

        return output_ids[assistant_role_slice.stop:]
    
    def get_llmloss(self,input_ids,suffix_manager,adv_suffix,batch_size):
        logits,ids= get_logits(model=self.model_wrapper.model, 
                                tokenizer=self.model_wrapper.tokenizer,
                                input_ids=input_ids,
                                control_slice=suffix_manager._control_slice, 
                                test_controls=adv_suffix, 
                                return_ids=True,
                                batch_size=batch_size)
        return target_loss(logits, ids, suffix_manager._target_slice)

    def check_for_attack_success(self,model, tokenizer, input_ids, assistant_role_slice, test_prefixes, gen_config=None):
        gen_str = tokenizer.decode(self.generate(model, 
                                            tokenizer, 
                                            input_ids, 
                                            assistant_role_slice, 
                                            gen_config=gen_config)).strip()
        jailbroken = not any([prefix in gen_str for prefix in test_prefixes])
        return jailbroken,gen_str
    

    def check_for_attack_success_and_log(self,adv_suffix,suffix_manager):
        # Notice that for the purpose of demo we stop immediately if we pass the checker but you are free to comment this to keep the optimization running for longer (to get a lower loss).

        self.input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(self.device)

        is_success,self.gen_str = self.check_for_attack_success(self.model_wrapper.model, 
                                self.model_wrapper.tokenizer,
                                self.input_ids,
                                suffix_manager._assistant_role_slice, 
                                self.test_prefixes)
        self.time_used = time.time()-self.time_start
        
        if self.change ==0: #record init state
            self.time_used = 0
            self.gen_str_init = self.gen_str
            self.loss = self.get_llmloss(self.input_ids,suffix_manager,[adv_suffix],batch_size=1).item()
            self.loss_init = self.loss

        logging.info('Current_origS: %s'%((adv_suffix)))
        logging.info('gen_str: %s'%((self.gen_str)))
        logging.info("changes: %s, loss: %s, time_used %s " %(self.change, self.loss,self.time_used))
        if self.args.wandb:
            commit = False if self.args.debug else True
            self.wandb_logger.log({'loss': self.loss,
                                    "change": self.change,
                                    'time_used': self.time_used}, commit=commit)
        # logging.info("\n\n\n\n")
        gc.collect()
        torch.cuda.empty_cache() 
        return is_success

    def attack(self,instruction,target,orig_S,test_prefixes,debug_init_loss=False,wandb_logger = None):
        self.test_prefixes = test_prefixes
        self.localsearch_z=None
        self.change = 0 # total change of character
        suffix_manager = SuffixManager(tokenizer=self.model_wrapper.tokenizer, 
                        conv_template=self.conv_template, 
                        instruction=instruction, 
                        target=target, 
                        adv_string=orig_S)
        
        self.time_start = time.time()
        if self.args.wandb:
            self.wandb_logger = wandb_logger
            self.wandb_logger.define_metric("change")
            self.wandb_logger.define_metric("time_used", step_metric="change")          
            self.wandb_logger.define_metric("loss", step_metric="change")
            self.wandb_logger.define_metric("pgd_axis")
            self.wandb_logger.define_metric("loss(mixed emb)", step_metric="pgd_axis")
            # self.wandb_logger.define_metric("table", step_metric="pgd_axis")
            # self.text_table = wandb.Table(columns=["epoch", "loss", "output(mixed emb)"])

        #check init loss
        is_success = self.check_for_attack_success_and_log(orig_S,suffix_manager)
        if debug_init_loss:
            return None
                    
        if self.args.attack_name == 'charmer' or self.args.n_positions!=-1:
            self.T_promptbegin = {}
            self.T_promptbegin['input_ids'] = self.input_ids[:suffix_manager._control_slice.start].unsqueeze(0)
            self.T_promptbegin['attention_mask'] = torch.ones_like(self.T_promptbegin['input_ids'])

            self.T_promptend = {}
            self.T_promptend['input_ids'] = self.input_ids[suffix_manager._assistant_role_slice].unsqueeze(0)
            self.T_promptend['attention_mask'] = torch.ones_like(self.T_promptend['input_ids'])

            self.T_label = {}
            self.T_label['input_ids'] =  self.input_ids[suffix_manager._assistant_role_slice.stop:].unsqueeze(0)
            self.T_label['attention_mask'] = torch.ones_like(self.T_label['input_ids'])
            
        # start iteratively attack
        with ctx_noparamgrad(self.model_wrapper.model): 
            for _ in range(self.args.k):
                self.change += 1
                # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
                input_ids = suffix_manager.get_input_ids(adv_string=orig_S).to(self.device)
                if self.args.attack_name == 'charmer':
                    if self.args.n_positions == -1:
                        subset_z = None
                    else:
                        subset_z = self.get_top_n_locations(orig_S,None)

                    SS,u,= self.attack_ours_pga(orig_S,subset_z = subset_z)

                    idxs = list(torch.topk(u.flatten(),5).indices)
                    new_sentences = [SS[j] for j in idxs]

                    #losses = self.get_llmloss_from_questions(self.criterion,new_sentences,T_questions=None)
                    losses= self.get_llmloss(input_ids,suffix_manager, new_sentences, batch_size=len(new_sentences))

                    id_best_batch = torch.argmax(-1 * losses, dim=0) #from 0 to 4
                    id_best = idxs[id_best_batch] #from 0 to len(SS)
                    orig_S = SS[id_best]
                    self.loss = losses[id_best_batch].item()
                    
                elif self.args.attack_name == 'bruteforce':
                    if self.args.n_positions == -1:
                        subset_z = None
                    else:
                        if 'v1' in self.args.llmselect:
                            subset_z = None
                            questions = utils.generate_all_sentences(orig_S,[ord(' ')], subset_z,1)
                            losses = self.get_llmloss(input_ids,suffix_manager, questions,batch_size=len(questions))
                            subset_z =  list(torch.topk(-1 * losses,self.args.n_positions).indices)                          
                        elif 'v4' in self.args.llmselect: # word level
                            topkword = self.args.topkword #3
                            #match word and idx
                            lasti = 0
                            questions,idlist,removeword= [],[],[]
                            for i,cha in enumerate(orig_S + ' '):
                                if cha == ' ':
                                    if lasti == 0:
                                        questions.append(orig_S[:lasti] + orig_S[i+1:])
                                        removeword.append(orig_S[lasti:i])
                                    else:
                                        questions.append(orig_S[:lasti] + orig_S[i:])
                                        removeword.append(orig_S[lasti+1:i])
                                    idlist.append(list(range(lasti*2,i*2)))
                                    lasti = i
                            #get loss
                            losses = self.get_llmloss(input_ids,suffix_manager, questions,batch_size=len(questions))
                            subset_z = []
                            for range_ in list(torch.topk(-1 * losses,topkword).indices):
                                subset_z+=idlist[range_]
                                logging.info('remove: '+ removeword[range_])
                        else:
                            raise NotImplementedError
                        if 'nosample' not in self.args.llmselect:
                            subset_z = subset_z + random.sample(
                                set(range((1+1)*len(orig_S) + 1))-set(subset_z),
                                self.args.n_positions) #changed n-position+random
                    SS,orig_S = self.attack_brute_force(input_ids,suffix_manager,orig_S,bs=self.args.bruteforce_batch,subset_z=subset_z)
                
                # elif self.args.attack_name == 'bruteforce_local':
                #     SS,u,losses = self.attack_brute_force(orig_S,bs=self.args.bruteforce_batch,greedynum = self.args.greedynum)
                #     id_best_batch = torch.topk(u.flatten(),1).indices

                #     for i,(a1,a2)  in enumerate(zip(SS[id_best_batch],orig_S)):
                #         if a1!=a2:
                #             localsearch_pos = i
                #             break
                #     self.localsearch_z =[2*localsearch_pos-2,
                #                          2*localsearch_pos-1,
                #                          2*localsearch_pos,
                #                          2*localsearch_pos+1]
                #     self.localsearch_z = [x for x in self.localsearch_z if x >= 0 ]
                #     orig_S = SS[id_best_batch]
                else:
                    raise NotImplementedError
                
                is_success = self.check_for_attack_success_and_log(orig_S,suffix_manager)
                if is_success:
                    break
        return orig_S,is_success
    

    def attack_ours_pga(self, S, subset_z = None, init = None):
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
                
        if self.args.combine_layer in ['encoder', 'encoder2']:
            # encoder_output, SS = self.perturbed_all(S,subset_z = subset_z)
            # encoder_output = encoder_output.detach()
            raise NotImplementedError
        elif self.args.combine_layer in ['emb', 'emb2']:
            emb_output, SS = self.perturbed_all(S,subset_z = subset_z)
            emb_output = emb_output.detach()
        else:
            raise NotImplementedError
        
        if init is None:
            u = torch.ones([len(SS),1]).to(self.device)/len(SS)
        else:
            u = init.to(self.device)
        
        lr = self.args.lr
        
        for i in range(self.args.n_iter):
            u.requires_grad = True
            
            if self.args.llm:                  
                if self.args.combine_layer in ['emb', 'emb2']:
                    combined_emb = (u.unsqueeze(-1).to(torch.float16)*emb_output).sum(dim=0, keepdim=True)
                    out = self.model_wrapper.emb_to_logit(inputs_embeds = combined_emb)
                else:
                    raise NotImplementedError
                
                loss_slice = slice(len(out[0]) - len(self.T_label['input_ids'][0])  - 1, len(out[0]) - 1)
                loss = self.criterion(out[0,loss_slice,:], self.T_label['input_ids'][0])
                if self.reg>0:
                    loss  = loss - 1 * self.reg * torch.norm(u, 1)
                
            loss.backward(retain_graph = True)
            g = u.grad.clone() 
            u = u.detach()
            u_hat = u.clone() + lr*g
            u = torch.tensor(self.simplex_projection(u_hat.flatten().tolist()), device = g.device).view(g.shape)

            lr*=self.args.decay
            
            if self.args.debug:
                if not self.args.llm:
                    logging.info(i, loss.item(), torch.topk(u.flatten(),5).values)
                    # logging.info('pred:', out, label)
                    # logging.info(g[:5])
                    # logging.info(torch.sum(torch.abs(u-u_hat)))
                else:
                    if i %2 ==0:
                        continue  
                    logging.info('loss(mixed emb) %s' %loss.item() )
                    logging.info('top u and sum %s %s'%(str(torch.topk(u.flatten(),5).values.cpu().numpy()),u.sum().item()))
                    #SEE GENERATE SENTENCE BY MIXED:
                    emb_without_label = combined_emb[:, :len(out[0]) - len(self.T_label['input_ids'][0]) ,:]
                    gen_str = self.model_wrapper.pred_by_generation_sentence(emb_without_label,self.args.generate_len).replace("\n", "")
                    logging.info('output(mixed emb): %s' %(gen_str)) 
                    # SEE GENERATE SENTENCE BY ARGMAX:
                    idxs = list(torch.topk(u.flatten(),5).indices)
                    new_sentence = SS[idxs[0]]
                    gen_str = self.model_wrapper.pred_by_generation_sentence([self.promptbegin+new_sentence+self.promptend],self.args.generate_len).replace("\n", "")
                    logging.info('ADVinput(argmax a): %s'%(new_sentence))
                    logging.info('Output(argmax a): %s'%( gen_str))

                    if self.args.wandb:
                        pgd_axis = self.change + (i/ self.args.n_iter)
                        # self.text_table.add_data(pgd_axis, loss.item(),gen_str) 
                        self.wandb_logger.log({"loss(mixed emb)" : loss.item(),
                                        # "table" : wandb.Table(columns=self.text_table.columns, data=self.text_table.data),
                                        "pgd_axis": pgd_axis},commit =True)
        return SS,u
    

    def attack_brute_force(self, input_ids,suffix_manager,S,bs = 100,subset_z=None):    
        with torch.no_grad():

            SS = utils.generate_all_sentences(S,self.V,subset_z = subset_z,k=1)
            SS = list(set(SS))
            # print("time_generate",time.time()-time_start)
            print(len(SS))
            if len(SS) > self.args.bruteforce_sample:
                SS = random.sample(SS, self.args.bruteforce_sample)
                if S not in SS:
                    SS[-1] = S
            else:
                if S not in SS:
                    SS.append(S)

            # if bs==-1 or bs>len(SS):
            #     bs = len(SS)
            # loss = []
            # for i in range(len(SS)//bs + (len(SS)%bs>0)):
            #     loss.append(self.get_llmloss(input_ids,
            #                                     suffix_manager,
            #                                     SS[i*bs: min((i+1)*bs, len(SS))],
            #                                     batch_size=bs))
            # loss = torch.cat(loss, dim=0)
            loss = self.get_llmloss(input_ids,suffix_manager, SS, batch_size=len(SS))

            id_best_batch = torch.topk(-1*loss.flatten(),1).indices
            orig_S = SS[id_best_batch]
            cur_loss = loss[id_best_batch].item()

            assert len(loss) == len(SS) and S in SS
            self.loss = cur_loss
        return SS,orig_S