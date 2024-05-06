#Code adapted from pytorch_transformers: https://github.com/huggingface/transformers
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json
import sys
import time
import pickle
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm, trange

from attacks import GreedyAttacker, BeamSearchAttacker, ED1AttackSurface
from recoverer import ClusterRepRecoverer, IdentityRecoverer, ScRNNRecoverer, ClusterIntprmRecoverer, RECOVERERS
from trans import TransformerRunner, ALL_MODELS, MODEL_CLASSES
from utils_roben import Clustering
from utils_glue import compute_metrics, GLUE_TASK_NAMES, OUTPUT_MODES, PROCESSORS
from augmentor import AUGMENTORS, IdentityAugmentor, HalfAugmentor, KAugmentor


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if args.seed_output_dir:
        print("Original output dir: ", args.output_dir)
        seed = int(time.time() * 100000)
        if args.output_dir.endswith('/'):
            args.output_dir = args.output_dir[:-1]
        print("Using seed {}".format(seed))
        args.output_dir = args.output_dir + '_' + str(seed)
        print("New output dir!: ", args.output_dir)


def save_results(args, results):
    save_dir = args.save_dir
    output_dirname = os.path.basename(os.path.normpath(args.output_dir))
    #Updated for codalab."
    filename = 'results.json'
    #filename = '{}_{}.json'.format(output_dirname, args.recoverer)
    results_dict = {'results': results, 'clusterer_path': args.clusterer_path, 'output_dir': args.output_dir, 'recoverer': args.recoverer, 
                    'num_epochs': args.num_train_epochs, 'attack_info': [args.attack, args.attacker, args.attack_name, args.beam_width],
                    'augmentor': args.augmentor, 'run_test': args.run_test}
    results_of_interest = ['acc', 'adv_acc', 'robust_acc']
    for res in results_of_interest:
        if res in results:
            results_dict[res] = results[res]
    print("About to save into: ", save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, 'w') as f:
        json.dump(results_dict, f)

def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(GLUE_TASK_NAMES))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", type=str, default = 'bert-base-uncased',
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--log_stdout_only', action='store_true',
                        help="Whether to log to stdout only")
    parser.add_argument('--verbose', action='store_true', help='Log verbosely')
    parser.add_argument('--save_steps', type=int, default=float('inf'),
                        help="Save checkpoint every X updates steps. Will save at the end regardless.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--clusterer_path', type = str, default = None, help = 
                        'Location of clusterer to be used, if clusterer is not None')

    parser.add_argument('--do_robust', action = 'store_true',
                        help = 'Compute robust accuracy (only tractable with clusters).')
    parser.add_argument('--robust_max_examples', type=int, default=10000,
                        help='Give up on certifying robustness if more than this many possibilities.')
    parser.add_argument('--print_examples', action = 'store_true',
                        help = 'Whether or not to print the adversarial worst case possibilities')
    parser.add_argument('--seed_output_dir', action = 'store_true',
                        help = 'Whether or not we should append something to the end of the output dir.')
    parser.add_argument('--save_results', action = 'store_true', help = 'Additionally save detailed results.')
    parser.add_argument('--save_dir', type = str, default = 'full_glue_run',
                        help = 'Directory to save the results of the run (within larger results_dir)')
    parser.add_argument('--recoverer', choices=RECOVERERS, default='identity',
                        help='Which recovery strategy to use (default: do nothing)')
    parser.add_argument('--augmentor', choices=AUGMENTORS, default = 'identity', help = 'How to augment data for training...')
    parser.add_argument('--lm_num_possibilities', type = int, default = 10,
                        help = 'Number of highest frequency options to consider per cluster in LM recovery (for efficiency)')
    parser.add_argument('--lm_ngram_size', type = int, default = 2,
                        help = 'Max size of n-grams in n-gram NgramLMRecoverer.')
    parser.add_argument('--error_analysis', action = 'store_true', help = 'Print out error analysis')
    parser.add_argument('--tc_dir', type = str, default = 'scRNN', 
                        help = 'Directory where typo-correctors and vocabs are stored (when recoverer is scrnn)')

    #Attack parameters
    parser.add_argument('--attack', action = 'store_true', help = 'Attack the clean model')
    parser.add_argument('--new_attack', action = 'store_true', help = 'Avoid loading from cached attack if it exists.')
    parser.add_argument('--attack_save_dir', type = str, default = 'attack_cache',
                        help = 'Location where the preprocessed attack files will be stored')
    parser.add_argument('--attack_name', type = str, default = 'DeleteAttack',
                        help = 'Name of the attack.')
    parser.add_argument('--attacker', type = str, choices = ['greedy', 'beam-search'], default = 'greedy',
                        help = 'Type of attack search strategy to use.')
    parser.add_argument('--attack_type', type = str, choices = ['ed1', 'intprm'], default = 'ed1',
                        help = 'Attack with edit distance one typos, or internal perturbations')
    parser.add_argument('--beam_width', type = int, default = 5, help = 'Width for beam search if used...')
    parser.add_argument('--analyze_res_attacks', action = 'store_true', 
                            help = 'Consider worst-case accuracy for different numbers of perturbations')
    parser.add_argument('--save_every_epoch', action = 'store_true',
                        help = 'Save checkpoints after every epoch, as opposed to just the last epoch...')
    parser.add_argument('--run_test', action = 'store_true', 
                    help = 'Evaluate on GLUE dev data, as opposed to a held out fraction of the training set.')
    parser.add_argument('--compute_ball_stats', action = 'store_true', help = 'Save the statistics about B_alpha')
    parser.add_argument('--compute_pred_stats', action = 'store', help = 'Store predictions for each rep...')
    return parser.parse_args()


def get_data(args):
    # Prepare GLUE task
    if args.task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = PROCESSORS[args.task_name]()
    augmentor = AUGMENTORS[args.augmentor]()
    output_mode = OUTPUT_MODES[args.task_name]
    label_list = processor.get_labels()
    train_data = processor.get_train_examples(args.data_dir)
    if args.run_test:
        dev_data = processor.get_dev_examples(args.data_dir)
    else:
        num_train_examples = int(len(train_data) * 0.8)
        dev_data = train_data[num_train_examples:]
        train_data = train_data[:num_train_examples]
    #Augmenting dataset
    train_data = augmentor.augment_dataset(train_data)
    args.output_mode = output_mode
    print("Train data len: {}, dev data len: {}".format(len(train_data), len(dev_data)))
    return train_data, dev_data, label_list

def get_recoverer(args):
    cache_dir = args.output_dir
    if args.recoverer == 'identity':
        return IdentityRecoverer(cache_dir)
    elif args.recoverer == 'scrnn':
        return ScRNNRecoverer(cache_dir, args.tc_dir, args.task_name)
    elif args.recoverer.startswith('clust'):
        clustering = Clustering.from_pickle(
                args.clusterer_path, max_num_possibilities=args.lm_num_possibilities)
        if args.recoverer == 'clust-rep':
            return ClusterRepRecoverer(cache_dir, clustering)
        elif args.recoverer == 'clust-intprm':
            return ClusterIntprmRecoverer(cache_dir, clustering)
    raise ValueError(args.recoverer)

def get_model_runner(args, recoverer, label_list):
    return TransformerRunner(
            recoverer, args.output_mode, label_list, args.output_dir, args.device, 
            args.task_name, args.model_type, args.model_name_or_path, 
            args.do_lower_case, args.max_seq_length)

def get_attacker(args, model_runner):
    if args.attacker == 'greedy':
        return GreedyAttacker(args.attack_name, args.task_name, model_runner,
                        args.attack_save_dir, args,)
    elif args.attacker == 'beam-search':
        print("Returning beam search attacker...")
        return BeamSearchAttacker(args.attack_name, args.task_name, model_runner,
                        args.attack_save_dir, args)
    else:
        raise ValueError(args.attacker)

def compute_ball_stats(dataset, model_runner, args, robust_max_examples = float('inf')):
    assert args.recoverer == 'clust-rep'
    assert args.run_test
    attack_surface = ED1AttackSurface()
    id2sizes = {}
    for ex in tqdm(dataset, desc = 'Getting example statistics'):
        text = ex.text_a
        if ex.text_b:
            text = '{} {}'.format(text, ex.text_b)
        clust_sizes, perturb_sizes = model_runner.recoverer.get_possible_recoveries(
                    text, attack_surface, robust_max_examples, ret_ball_stats = True)
        id2sizes[ex.guid] = clust_sizes, perturb_sizes 
    return id2sizes

def save_ball_stats(args, ball_stats):
    task = args.task_name
    ids = list(ball_stats)
    clusterer_name = os.path.basename(args.clusterer_path)
    if not os.path.exists('results/stats'):
        os.mkdir('results/stats')
    task_dir = os.path.join('results/stats', task)
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
    stats_dir = os.path.join(task_dir, clusterer_name)
    if not os.path.exists(stats_dir):
        os.mkdir(stats_dir)
    clust_stats_fn = os.path.join(stats_dir, 'ball_stats.txt')
    pert_stats_fn = os.path.join(stats_dir, 'perturbation_stats.txt')
    clust_lines = []
    perturb_lines = []
    for guid in ids:
        clust_sizes, pert_sizes = ball_stats[guid]
        clust_size_str = ','.join([str(size) for size in clust_sizes])
        pert_size_str = ','.join([str(size) for size in pert_sizes])
        clust_line = '{}:{}\n'.format(guid, clust_size_str)
        pert_line = '{}:{}\n'.format(guid, pert_size_str)
        clust_lines.append(clust_line)
        perturb_lines.append(pert_line)
    with open(clust_stats_fn, 'w') as f:
        f.writelines(clust_lines)
    with open(pert_stats_fn, 'w') as f:
        f.writelines(perturb_lines)

def evaluate(model_runner, dataset, batch_size, do_robust=False, robust_max_examples=10000, analyze_res_attacks = False):
    # Need to run recoverer manually on queries 
    # Because we're using recoverer.get_possible_examples
    all_examples = [model_runner.recoverer.recover_example(ex) for ex in tqdm(dataset, desc='Recovering dev')]

    if do_robust:
        num_reps = defaultdict(lambda: 0)
        attack_surface = ED1AttackSurface()
        id_to_poss_ids = {}
        num_poss_list = []
        num_exceed_max = 0
        for ex in tqdm(dataset, desc='Getting possible recoveries'):
            poss_exs, num_poss = model_runner.recoverer.get_possible_examples(
                    ex, attack_surface, robust_max_examples, analyze_res_attacks = analyze_res_attacks)
            num_reps[num_poss] += 1
            num_poss_list.append(num_poss)
            if poss_exs:
                all_examples.extend(poss_exs)
                id_to_poss_ids[ex.guid] = [x.guid for x in poss_exs]
            else:
                num_exceed_max += 1
        median = sorted(num_poss_list)[int(len(num_poss_list) / 2)]
        num_poss_under_thresh = [x for x in num_poss_list if x <= robust_max_examples]
        avg_under_thresh = sum(num_poss_under_thresh) / len(num_poss_under_thresh)
        print('Robust eval: %d median poss/ex; %d/%d exceed max of %d; %.1f avg poss/ex on remainder' % ( 
                median, num_exceed_max, len(dataset), robust_max_examples, avg_under_thresh))
    num_correct = 0
    if do_robust:
        num_robust = 0
    preds = model_runner.query(all_examples, batch_size, do_evaluate=not do_robust, do_recover=False)
    id_to_pred = {all_examples[i].guid: preds[i] for i in range(len(all_examples))}
    num_with_incorrect = 0
    assert(len(id_to_pred) == len(all_examples))
    if analyze_res_attacks:
        maxp2ncorrect = defaultdict(lambda: 0)
    for ex in dataset:
        pred = id_to_pred[ex.guid]
        was_correct = False
        if pred == ex.label:
            was_correct = True
            num_correct += 1
        if do_robust:
            if ex.guid not in id_to_poss_ids:
                continue
            poss_ids = id_to_poss_ids[ex.guid]
            cur_preds = set([id_to_pred[pid] for pid in poss_ids])
            if analyze_res_attacks:
                incorrect_preds = [int(pid.split('-')[3]) for pid in poss_ids if id_to_pred[pid] != ex.label]
                zero_count = len([pred for pred in incorrect_preds if pred == 0])
                assert zero_count < 2
                if len(incorrect_preds) != 0:
                    num_with_incorrect += 1
                    #Only adding things that were wrong initially
                    min_required_changes = min(incorrect_preds)
                    if was_correct:
                        if min_required_changes == 0:
                            print(incorrect_preds)
                            print(poss_ids)
                            print([id_to_pred[poss_id] for poss_id in poss_ids])
                            print(ex.guid, id_to_pred[ex.guid])
                            print("Label: ", ex.label)
                            print("Cur preds: ", cur_preds)
                            assert False

                    #Adds things where an incorrect prediction was found...
                    for n_changes in range(min_required_changes):
                        maxp2ncorrect[n_changes] += 1

            if all(x == ex.label for x in cur_preds):
                num_robust += 1
    print('Normal accuracy: %d/%d = %.2f%%' % (num_correct, len(dataset), 100.0 * num_correct / len(dataset)))
    results = {'acc': num_correct / len(dataset)}
    if do_robust:
        print('Robust accuracy: %d/%d = %.2f%%' % (num_robust, len(dataset), 100.0 * num_robust / len(dataset)))
        results['robust_acc'] = num_robust / len(dataset)
        if analyze_res_attacks:
            print(maxp2ncorrect)
            for n_changes in maxp2ncorrect:
                maxp2ncorrect[n_changes] = (maxp2ncorrect[n_changes] + num_robust) / len(dataset)
            maxp2ncorrect[len(maxp2ncorrect)] = num_robust / len(dataset)
            maxp2ncorrect = dict(maxp2ncorrect)
            results['restricted_acc_dict'] = maxp2ncorrect
            print(maxp2ncorrect)
        results['avg_under_thresh'] = avg_under_thresh
        results['median_under_thresh'] = median
        results['num_exceed_max'] = num_exceed_max
        #results['num_rep_analysis'] = dict(num_reps)
        print(dict(num_reps))

    return results

def main():
    args = parse_args()
    args.task_name = args.task_name.lower()
    if args.save_results:
        if not args.do_eval:
            raise ValueError("Must evaluate to save results (i.e. use --do_eval)")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    set_seed(args)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.log_stdout_only:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    else:
        logging.basicConfig(filename=os.path.join(args.output_dir, 'log.txt'), level=logging.DEBUG)

    # Get data and model
    train_data, dev_data, label_list = get_data(args)
    recoverer = get_recoverer(args) 
    model_runner = get_model_runner(args, recoverer, label_list)
    logger.info("Training/evaluation parameters %s", args)

    # Run training and evaluation
    if args.do_train:
        train_recovered = [recoverer.recover_example(x) for x in tqdm(train_data, desc='Recovering train')]
        model_runner.train(train_recovered, args)
    if args.compute_ball_stats:
        ball_stats_dict = compute_ball_stats(dev_data, model_runner, args, robust_max_examples = 10000)
        save_ball_stats(args, ball_stats_dict)
    if args.do_eval:
        results = evaluate(model_runner, dev_data, args.eval_batch_size, 
                           do_robust=args.do_robust, robust_max_examples=args.robust_max_examples,
                           analyze_res_attacks = args.analyze_res_attacks)
        if args.attack:
            attacker = get_attacker(args, model_runner)
            adv_data = attacker.attack_dataset(dev_data)
            print('Running adversarial evaluation.')
            adv_results = evaluate(model_runner, adv_data, args.eval_batch_size)
            for k, v in adv_results.items():
                results['adv_{}'.format(k)] = v
        print('Results: {}'.format(json.dumps(results)))
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f)
        if args.save_results:
            save_results(args, results)
    #recoverer.save_cache()

if __name__ == "__main__":
    main()
