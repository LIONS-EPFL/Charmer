import pandas as pd
import os
from tqdm import tqdm

from utils_glue import PROCESSORS
import argparse
import sys
import csv
csv.field_size_limit(sys.maxsize//10)


#Script used to convert data to something the scRNN can train...

def preprocess_for_typo_corrector(task, glue_data_dir, tc_preprocess_data_dir):
    print("Peprocessing for {}".format(task))
    task_data_dir = os.path.join(glue_data_dir, task)
    task = task.lower()
    processor = PROCESSORS[task]()
    train_examples = processor.get_train_examples(task_data_dir)
    has_b = train_examples[0].text_b is not None
    example_dicts = []
    for example in tqdm(train_examples):
        example_dict = {}
        example_dict['text_a'] = example.text_a
        if has_b:
            example_dict['text_b'] = example.text_b
        example_dicts.append(example_dict)
    data = pd.DataFrame(example_dicts)
    save_name = os.path.join(tc_preprocess_data_dir, '{}_train_preprocessed.tsv'.format(task))
    if not os.path.exists(tc_preprocess_data_dir):
        os.mkdir(tc_preprocess_data_dir)
    data.to_csv(save_name, sep = '\t')
    print ("Saved at {}".format(save_name))

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--glue_dir', type = str, default = 'data/glue_data',
                        help = 'Directory where glue data is stored')
  parser.add_argument('--save_dir', type = str, default = 'data/glue_tc_preprocessed',
                        help = 'Directory where preprocessed glue data will be stored.')
  return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    glue_data_dir = args.glue_dir
    tc_preprocess_data_dir = args.save_dir
    #tasks = ['SST-2', 'MRPC', 'QQP', 'MNLI', 'QNLI', 'RTE']
    tasks = ['SST-2','RTE', 'MNLI', 'QNLI']
    for task in tasks:
        preprocess_for_typo_corrector(task, glue_data_dir, tc_preprocess_data_dir)
