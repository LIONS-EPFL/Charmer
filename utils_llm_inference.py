# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# import openai
from baseline.llm.promptbench.config import LABEL_TO_ID,LABEL_SET 
from tqdm import tqdm
import pdb
import utils
import copy
import torch
import gc
"""
This clss implements the inference of the model (including create the model).
"""
class Inference(object):

    def __init__(self,args):
        self.error_analysis = False
        self.args = copy.copy(args)
        self.device = args.device
        self.model_name = args.model_name
        if self.args.dataset == 'sst': 
            self.args.dataset = 'sst2'
        self.create_model()

    def create_model(self):
        """
        ChatGPT is a special case, we use the openai api to create the model.
        """

        if self.model_name not in ['chatgpt', 'gpt4']:
            import torch
            import os
            
            """
            Here you can add you own model.
            """

            if self.model_name == 'google/flan-t5-large':
                from transformers import T5Tokenizer, T5ForConditionalGeneration

                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

            elif self.model_name == 'EleutherAI/gpt-neox-20b':
                from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
                
                self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(self.model_name, device_map="auto")
                self.model = GPTNeoXForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.float16)
 
            elif 'llama' in self.model_name:
                from transformers import AutoModelForCausalLM,AutoTokenizer
                self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir = "/home/data_shared/llama/",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True, 
                        use_cache=False).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name,
                            cache_dir = "/home/data_shared/llama/",
                            trust_remote_code=True,
                            use_fast=False)
                self.tokenizer.pad_token = self.tokenizer.unk_token
                self.tokenizer.padding_side = 'left' #this is important! not right!
                
            elif self.model_name.lower() in ["vicuna-13b", "vicuna-13b-v1.3"]:

                from transformers import AutoModelForCausalLM, AutoTokenizer

                model_dir = os.path.join(self.args.model_dir, self.model_name)

                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", use_fast=False)
                self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)
            elif 'vicuna-7b-v1.3' in self.model_name:
                from transformers import AutoModelForCausalLM,AutoTokenizer
                cache_dir = "/home/data_shared/vicuna/"
                self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir = cache_dir,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        use_cache=False
                    ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir = cache_dir,
                    trust_remote_code=True,
                    use_fast=False)
                self.tokenizer.pad_token = self.tokenizer.unk_token
                # self.tokenizer.padding_side = 'left' #this is important! not right! #TODO, why gcg didn't have this
            elif 'guanaco' in self.model_name:
                from transformers import AutoModelForCausalLM,AutoTokenizer
                cache_dir = "/home/data_shared/guanaco/" 
                self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir = cache_dir,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        use_cache=False
                    ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir = cache_dir,
                    trust_remote_code=True,
                    use_fast=False)
                self.tokenizer.pad_token = self.tokenizer.unk_token
                #adapt from 
                self.tokenizer.eos_token_id = 2
                self.tokenizer.unk_token_id = 0
                if not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            elif self.model_name == "google/flan-ul2":

                from transformers import T5ForConditionalGeneration, AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto")

            elif self.model_name == "tiiuae/falcon-40b-instruct":                                                         
                from transformers import AutoTokenizer, AutoModelForCausalLM

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",)
                

            elif self.model_name == "cerebras/Cerebras-GPT-13B":
                from transformers import AutoTokenizer, AutoModelForCausalLM

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.float16)

            elif self.model_name == "databricks/dolly-v1-6b":
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v1-6b", device_map="auto", padding_side="left")
                self.model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v1-6b", device_map="auto", torch_dtype=torch.float16)
            
            else:
                raise NotImplementedError("The model is not implemented!")

    def process_input(self, prompt, raw_data):
        if self.args.dataset in ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]:
            return self._process_cls_input(prompt, raw_data)
        elif self.args.dataset == "mmlu":
            return self._process_qa_input(prompt, raw_data)
        elif self.args.dataset == "squad_v2":
            return self._process_squad_v2_input(prompt, raw_data)
        elif self.args.dataset in ['iwslt', 'un_multi']:
            return self._process_trans_input(prompt, raw_data)
        elif self.args.dataset == 'math':
            return self._process_math_input(prompt, raw_data)
        elif self.args.dataset == 'bool_logic':
            return self._process_bool_logic_input(prompt, raw_data)
        elif self.args.dataset == 'valid_parentheses':
            return self._process_valid_parentheses_input(prompt, raw_data)
        else:
            raise NotImplementedError("The dataset is not implemented!")
    
    def process_pred(self, pred):
        if self.args.dataset in ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]:
            return self._process_cls_pred(pred)
        elif self.args.dataset == "mmlu":
            return self._process_qa_pred(pred)
        elif self.args.dataset == "squad_v2":
            return self._process_squad_v2_pred(pred)
        elif self.args.dataset in ['iwslt', 'un_multi']:
            return self._process_trans_pred(pred)
        elif self.args.dataset == 'math':
            return self._process_math_pred(pred)
        elif self.args.dataset == 'bool_logic':
            return self._process_bool_logic_pred(pred)
        elif self.args.dataset == 'valid_parentheses':
            return self._process_valid_parentheses_pred(pred)
        else:
            raise NotImplementedError("The dataset is not implemented!")

    def __call__(self, SS, debug=False):
        preds = []
        for S in SS:
            if isinstance(S, tuple) or isinstance(S, list):
                premise = S[0]
                S = S[1]
            else:
                premise = None
            llm_input_processor = utils.llm_input_processor(self.args.dataset,premise,self.args.model_name)
            S =  llm_input_processor.addprompt(S)
            if self.model_name in ["chatgpt", "gpt4"]:
                pred = self.predict_by_openai_api(S)
            else:
                pred= self.predict_by_local_inference(S,debug)
            
            if len(SS)!=1:
                pred = pred.detach().cpu() 
            preds.append(pred)
        return preds

    def predict_by_openai_api(self, S):
        raise NotImplementedError

    def predict_by_local_inference(self, S,debug=False):
        "return  logit with size (batch,vocabularysize)"
        logits = self.pred_by_generation(S)
        if debug:
            print('orig_S:', S)
            idx = torch.argmax(logits)
            raw_pred = self.tokenizer.decode(idx, skip_special_tokens=True)
            print('raw_pred:',raw_pred)
        return logits

    def token_to_logit(self,input_ids,attention_mask = None):
        '''
        return  logit with size (batch,len,vocabularysize)
        '''
        if 'llama' in self.model_name or 'vicuna-7b-v1.3' in self.model_name:
            pred = self.model(input_ids =input_ids, attention_mask =attention_mask).logits
        elif 't5' in self.model_name:
            inputs_embeds = self.model.encoder.embed_tokens(input_ids)
            pred = self.emb_to_logit(inputs_embeds,attention_mask)
        else: 
            raise NotImplementedError

        return pred
        
    def emb_to_logit(self,inputs_embeds,attention_mask=None):    
        '''
        return  logit with size (batch,len,vocabularysize)
        '''
        if 'llama' in self.model_name or 'vicuna-7b-v1.3' in self.model_name:
            out = self.model(inputs_embeds = inputs_embeds).logits
        
        elif 't5' in self.model_name:
            decoder_input_ids = torch.ones(inputs_embeds.shape[0], 1,dtype=torch.int) \
                * self.model.config.decoder_start_token_id
            decoder_inputs_embeds = self.model.decoder.embed_tokens(decoder_input_ids.to(self.device))
            
            out = self.model(inputs_embeds=inputs_embeds,
                            decoder_inputs_embeds=decoder_inputs_embeds,
                            attention_mask = attention_mask).logits
            # TODO check attention mask here
        else: 
            raise NotImplementedError
        return out

    def pred_by_generation_sentence(self, inputs,generate_len):
        '''
        return  sentence in jailbreak
        '''
        gen_config  = self.model.generation_config
        gen_config.max_new_tokens = generate_len # self.args.generate_len
        
        if isinstance(inputs[0], str):
            # str to str
            input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids.to(self.device)
            outputs = self.model.generate(input_ids, 
                                        generation_config=gen_config,
                                        pad_token_id=self.tokenizer.pad_token_id)
            out = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            return out
            # else:
            #     raise NotImplementedError
        elif torch.is_tensor(inputs):
            # emb to str
            outputs = self.model.generate(inputs_embeds = inputs, 
                                        generation_config=gen_config,
                                        pad_token_id=self.tokenizer.pad_token_id)
            out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return out
            # else:
            #     raise NotImplementedError
        else:
            raise NotImplementedError
    
    def pred_by_generation(self, input_text):
        '''
        return  logit with size (batch,vocabularysize)
        '''

        out = 'error!'
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        if 't5' in self.model_name or 'ul2' in self.model_name:
            inputs_embeds = self.model.encoder.embed_tokens(input_ids)
            out = self.emb_to_logit(inputs_embeds)
            out = out[0,-1,:]
            return out

        elif self.model_name == 'EleutherAI/gpt-neox-20b':
            outputs = self.model.generate(input_ids, 
                                        #  do_sample=True, 
                                         temperature=0.00001, 
                                        #  max_length=50,
                                         max_new_tokens=self.args.generate_len,
                                         early_stopping=True,
                                         pad_token_id=self.tokenizer.eos_token_id)
            
            out = self.tokenizer.decode(outputs[0])

        elif self.model_name == "facebook/opt-66b":
            outputs = self.model.generate(input_ids)
            out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elif 'llama' in self.model_name or 'vicuna-7b-v1.3' in self.model_name:
            out = self.model(input_ids).logits
            out = out[0,-1,:]
            return out
      
        elif self.model_name in ['databricks/dolly-v1-6b', 'cerebras/Cerebras-GPT-13B']:
            outputs = self.model.generate(input_ids, 
                                         temperature=0,
                                         max_new_tokens=self.args.generate_len,
                                         pad_token_id=self.tokenizer.eos_token_id,
                                         early_stopping=True)
            
            out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elif self.model_name == "tiiuae/falcon-40b-instruct":
            outputs = self.model.generate(input_ids, 
                                         temperature=0,
                                         max_new_tokens=self.args.generate_len,
                                         pad_token_id=self.tokenizer.eos_token_id,
                                         early_stopping=True)
            
            out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return out

    def _process_valid_parentheses_input(self, prompt, raw_data):
        question, label = raw_data['question'], raw_data['answer']
        input_text = prompt + '\n'
        
        if self.args.shot > 0:
            input_text += "\n"+self.args.data.get_few_shot_examples(raw_data['task'])
        
        input_text += ("Question: " + question + '\nAnswer: ')

        return input_text, label

    def _process_bool_logic_input(self, prompt, raw_data):
        question, label = raw_data['question'], raw_data['answer']
        input_text = prompt + '\n'
        
        if self.args.shot > 0:
            input_text += "\n"+self.args.data.get_few_shot_examples(raw_data['task'])
        
        input_text += ("Question: " + question + '\nAnswer: ')

        return input_text, label

    def _process_math_input(self, prompt, raw_data):
        from config import MATH_QUESTION_TYPES
        question_type, question, label = MATH_QUESTION_TYPES[raw_data['task']], raw_data['question'], raw_data['answer']
        input_text = prompt.format(question_type) + '\n'
        
        if self.args.shot > 0:
            input_text += "\n"+self.args.data.get_few_shot_examples(raw_data['task'])
        
        input_text += ("Question: " + question + '\nAnswer: ')

        return input_text, label

    def _process_trans_input(self, prompt, raw_data):
        from config import LANGUAGES
        source, target, task = raw_data['source'], raw_data['target'], raw_data['task']
        src_lang, des_lang = task.split('-')
        input_text = prompt.format(LANGUAGES[src_lang], LANGUAGES[des_lang]) + '\n'

        if self.args.shot > 0:
            input_text += "\n"+self.args.data.get_few_shot_examples(task)

        input_text += (source + '\nAnswer: ')
        return input_text, target

    def _process_squad_v2_input(self, prompt, raw_data):
        id, content = raw_data["id"], raw_data["content"]
        input_text = prompt

        if self.args.shot > 0:
            input_text += "\n"+self.args.data.get_few_shot_examples(self.args.dataset)

        input_text += (content + "Answer: ")

        return input_text, id

    def _process_qa_input(self, prompt, raw_data):
        task, content = raw_data["task"], raw_data["content"]
        label = raw_data["label"]

        input_text = prompt.format(task) + "\n"

        if self.args.shot > 0:
            input_text += "\n"+self.args.data.get_few_shot_examples(task.replace(" ", "_"))
        
        input_text += content + "\nAnswer: "
        
        return input_text, label

    def _process_cls_input(self, prompt, raw_data):
        content = raw_data["content"]
        label = raw_data["label"]

        input_text = prompt

        if self.args.shot > 0:
            few_shot_examples = self.args.data.get_few_shot_examples(self.args.dataset)
            input_text += "\n"+few_shot_examples
            if  self.args.dataset == "sst2" or  self.args.dataset == "cola":
                input_text += "Sentence: "
          
        input_text += (content + ' Answer: ')

        return input_text, label

    def _process_bool_logic_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        return pred
    
    def _process_valid_parentheses_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        return pred
    
    def _process_math_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        return pred
    
    def _process_trans_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        return pred

    def _process_squad_v2_pred(self, raw_pred):
        pred = raw_pred.lower()
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        return pred

    def _process_cls_pred(self, raw_pred):

        pred = raw_pred.lower()

        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
        if self.args.dataset!= 'sst2':
            pred = pred.split(" ")[-1]
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")

        if pred in LABEL_SET[self.args.dataset]:                   
            pred = LABEL_TO_ID[self.args.dataset][pred] 
        else:             
            print("The predicted label: '{}' is not in label set.".format(pred))
            pred = -1
        
        return pred
        
    def _process_qa_pred(self, raw_pred):
        pred = raw_pred.lower()
        
        pred = pred.replace("<pad>", "")
        pred = pred.replace("</s>", "")
        
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
        pred = pred.split(" ")[-1]
        pred = pred.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
        
        if pred not in LABEL_SET[self.args.dataset]:   
            print("The predicted label: '{}' is not in label set.".format(pred))
            pred = 'no_answer'

        return pred
    