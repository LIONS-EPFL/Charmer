# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

MNLI_LABEL = ['entailment', 'neutral', 'contradiction', 'entailment\'', 'neutral\'', 'contradiction\'']
EQ_LABEL = ['equivalent', 'not_equivalent', 'equivalent\'', 'not_equivalent\'']
# ENTAIL_LABEL = ['entailment', 'not_entailment', 'entailment\'', 'not_entailment\'', '0', '1', '0\'', '1\'', 0, 1]
ENTAIL_LABEL = ['entailment', 'not_entailment', 'entailment\'', 'not_entailment\'', '0', '1', '0\'', '1\'']  #changed! there is a bug

LABEL_SET = {
    # 'positive\'', 'negative\'' is used for label constraint due to a bug of TextAttack repo.
    
    # 'sst2': ['positive', 'negative', 'positive\'', 'negative\'', '0', '1', '0\'', '1\'',0, 1],  #changed! there is a bug
    'sst2': ['positive', 'negative', 'positive\'', 'negative\'', '0', '1', '0\'', '1\''], 
    'mnli': MNLI_LABEL,
    'mnli_mismatched': MNLI_LABEL,
    'mnli_matched': MNLI_LABEL,
    'qqp': EQ_LABEL,
    'qnli': ENTAIL_LABEL,
    'rte': ENTAIL_LABEL,
    'cola': ['unacceptable', 'acceptable', 'unacceptable\'', 'acceptable\''],
    'mrpc': EQ_LABEL,
    'wnli': ENTAIL_LABEL,
    'mmlu': ['A', 'B', 'C', 'D', 'A\'', 'B\'', 'C\'', 'D\'', 'a', 'b', 'c', 'd', 'a\'', 'b\'', 'c\'', 'd\''],
    'squad_v2': ['unanswerable', 'unanswerable\''], # do not change the word 'nothing' in prompts.
    'iwslt': ['translate', 'translate\''],
    'un_multi': ['translate', 'translate\''],
    'math': ['math', 'math\''],
    'bool_logic': ['True', 'False', 'True\'', 'False\'', "bool", "boolean", "bool\'", "boolean\'"],
    'valid_parentheses': ['Valid', 'Invalid', 'Valid\'', 'Invalid\'', 'matched', 'matched\'', 'valid', 'invalid', 'valid\'', 'invalid\''],
}

GENERATE_LEN = {
    'sst2': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 2, 'llama2-13b': 2,},
    'mnli': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 3, 'llama2-13b': 3,},
    'qqp': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 3, 'llama2-13b': 4},
    'qnli': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 4, 'llama2-13b': 2,},
    'rte': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 4, 'llama2-13b': 3,},
    'cola': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 3, 'llama2-13b': 3,},
    'mrpc': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 3, 'llama2-13b': 2,},
    'wnli': {'google/flan-t5-large': 20, 'vicuna-13b': 5, 'google/flan-ul2': 20, "chatgpt": 4, 'llama2-13b': 3,},
    'mmlu': {'google/flan-t5-large': 2, 'vicuna-13b': 2, 'google/flan-ul2': 2, "chatgpt": 2, 'llama2-13b': 3,},
    'squad_v2': {'google/flan-t5-large': 20, 'google/flan-ul2': 20, "chatgpt": 20},
    'iwslt': {'google/flan-t5-large': 70, 'google/flan-ul2': 70, 'chatgpt': 70},
    'un_multi': {'google/flan-t5-large': 140, 'google/flan-ul2': 140, 'chatgpt': 140},
    'math': {'google/flan-t5-large': 20, 'google/flan-ul2': 20, 'chatgpt': 20},
    'bool_logic': {'google/flan-t5-large': 4,},
}

MODEL_SET = [ 
    'google/flan-t5-large',
    'EleutherAI/gpt-neox-20b',
    'tiiuae/falcon-40b-instruct',
    # 'facebook/opt-66b', 
    'llama-13b',
    'llama2-13b',
    'llama2-13b-chat',
    'vicuna-13b',
    'vicuna-13b-v1.3',
    'google/flan-ul2',
    'cerebras/Cerebras-GPT-13B',
    'databricks/dolly-v1-6b',
    'chatgpt',
    'gpt4'
]

LABEL_TO_ID = {
    'mmlu': {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'},
    'sst2': {'negative': 0, 'positive': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'mnli': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2':2, 0: 0, 1: 1, 2: 2},
    'mnli_mismatched': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2':2, 0: 0, 1: 1, 2: 2},
    'mnli_matched': {'entailment': 0, 'neutral': 1, 'contradiction': 2, '0': 0, '1': 1, '2':2, 0: 0, 1: 1, 2: 2},
    'qqp': {'equivalent': 1, 'not_equivalent': 0, '0': 0, '1': 1, 0: 0, 1: 1},
    'qnli': {'entailment': 0, 'not_entailment': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'rte': {'entailment': 0, 'not_entailment': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'cola': {'unacceptable': 0, 'acceptable': 1, '0': 0, '1': 1, 0: 0, 1: 1},
    'mrpc': {'equivalent': 1, 'not_equivalent': 0, '0': 0, '1': 1, 0: 0, 1: 1},
    'wnli': {'entailment': 1, 'not_entailment': 0, '0': 0, '1': 1, 0: 0, 1: 1},
}

ID_TO_LABEL = {
    'mmlu': {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'},
    'sst2': {0: 'negative', 1: 'positive'},
    'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mnli_matched': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mnli_mismatched': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'qqp': {1: 'equivalent', 0: 'not_equivalent'},
    # 'qnli': {0: 'entailment', 1: 'not_entailment'},
    'qnli': {0: 'yes', 1: 'no'}, #changed, because our method only supports one-token output
    # 'rte': {0: 'entailment', 1: 'not_entailment'},
    'rte': {0: 'yes', 1: 'no'}, #changed, because our method only supports one-token output
    'cola': {0: 'unacceptable', 1: 'acceptable'},
    'mrpc': {1: 'equivalent', 0: 'not_equivalent'},
    'wnli': {1: 'entailment', 0: 'not_entailment'},
}

SUPPORTED_LANGUAGES = {
    'google/flan-t5-large': ['en', 'de', 'fr'],
    'google/flan-ul2': ['en', 'de', 'fr'],
    'vicuna-13b': ['en', 'de', 'fr'],
    'llama2-13b-chat': ['en', 'de', 'fr'],
    'chatgpt': ['en', 'de', 'fr'],
}

LANGUAGES = {
    'ar': 'Arabic', 
    'de': 'German', 
    'en': 'English', 
    'es': 'Spanish',
    'fr': 'French',
    'ru': 'Russian',
    'zh': 'Chinese',
    'it': 'Italian',
    'nl': 'Dutch',
    'ro': 'Romanian',
    'ja': 'Japanese',
    'ko': 'Korean',
}

MATH_QUESTION_TYPES = {
    'algebra_linear_1d': ' linear algebra ',
    'algebra_linear_2d': ' linear algebra ',
    'algebra_sequence_next_term': ' given a sequence predict the next term ',
    'arithmetic_addition_sub_multiple': ' arithmetic addition and subtraction ',
    'arithmetic_mul_div_multiple': ' arithmetic multiplication and division ',
    'arithmetic_mixed': ' arithmetic addition, subtraction, multiplication and division ',
    'arithmetic_nearest_integer_root': ' arithmetic nearest integer root ',
    'comparison_closest': ' compare which one of given numbers is closest to target number ',
    'comparison_kth_biggest': ' compare which one of given numbers is kth biggest or smallest ',
    'comparison_pair': ' comparison which one of given numbers is bigger or smaller ', 
    'measurement_conversion': ' measurement conversion ',
    'numbers_base_conversion': ' numbers base conversion ',
    'numbers_div_remainder': ' numbers division and remainder ',
    'numbers_gcd': ' numbers greatest common divisor ',
    'numbers_is_factor': ' if one number is a factor of antoher number ',
    'number_is_prime': ' if a number is prime ',
    'numbers_lcm': ' least common multiple ',
    'numbers_place_value': ' place value ',
    'numbers_round_number': ' round number ',
    'polynomials_evaluate': ' polynomials evaluate ',
}