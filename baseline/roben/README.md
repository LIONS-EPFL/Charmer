Code for the following paper:
> [Robust Encodings: A Framework for Combating Adversarial Typos](https://arxiv.org/abs/2005.01229)
>
> Erik Jones, Robin Jia, Aditi Raghunathan, and Percy Liang
>
> Association for Computational Linguistics (ACL), 2020

## Cluster Embeddings
We will run experiments for six tasks: RTE, MRPC, SST-2, QNLI, MNLI, QQP. These are used as arguments whenever task name (or mrpc in the following code, which is used as an example) comes up. Data is available on codalab.
### Standard training 
The core element of our defense is a "clusterer" object, which we use to map tokens to a series of representatives, before inputting into a normal model. To create a clusterer, we use two different data sources: 
* Embeddings used to filter vocab words: ```data/glove/glove.6b.50d.txt```
* Word frequencies: ```data/COCA/coca-1grams.json```
Given these files, to make a clusterer, run:
```python construct_clusters.py --vocab_size 100000 --perturb_type ed1```
This will form a clusterer object with path ```clusterers/vocab100000_ed1.pkl```, which will be used in future experiments.

Now, lots of the following code is adapted from an older version of https://github.com/huggingface/transformers. Data can be found there. We will first fine-tune and save uncased BERT on the MRPC task. To do so, we set the following variables:
```
export TASK_NAME=MRPC
export CLUSTERER_PATH=clusterers/vocab100000_ed1.pkl
export GLUE_DIR=data/glue_data
```
Where the data from MRPC is stored in ```glue_data``` With these variables set, we run:
```
python run_glue.py --task_name $TASK_NAME --do_lower_case --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME --output_dir model_output/$TASK_NAME --overwrite_output_dir --seed_output_dir --save_results --save_dir codalab --recoverer identity --augmentor identity --run_test
```
This gives us a normally trained model, which will get saved at model_output/MRPC_XXXXXX where XXXXXX is a random six digit number (this is the ```--seed_output_dir``` argument. Information (including clean accuracy which we report, and future attack statistics) will be stored in results/codalab/MRPC_XXXXXX.json. To attack this model, we run: 
```
python run_glue.py --task_name $TASK_NAME --do_lower_case --do_eval --data_dir $GLUE_DIR/$TASK_NAME --output_dir model_output/$TASK_NAME_XXXXXX  --save_results --save_dir codalab --recoverer identity --augmentor identity --run_test --model_name_or_path model_output/MRPC_XXXXXX --attack --new_attack --attacker beam-search --beam_width 5 --attack_name LongDeleteShortAll --attack_type ed1
```
There are a lot of arguments here. ```attack``` means an adversary is searching for a typo, and ```new_attack``` says to avoid a cache. ```attacker``` determines the style of heuristic attack, and ```attack_name``` gives the type of token-level peturbation space used for the attack. This is all the information we need for the identity. 
### Data Augmentation
To run this experiment with data augmentation, repeat both runs of python run_glue.py, but with the flag ```--augmentor k-aug```.

### Typo Corrector
We'll now replicate the entire typo corrector training process, utilizing the new environment variable:
```
$TC_DIR=$HOME/tc_data
```
This will have to be made if it does not exist, but it will store preprocessed data, vocabularies, and models. First, we run: 
```
preprocess_tc.py --glue_dir $GLUE_DIR --save_dir $TC_DIR/glue_tc_preprocessed
```
This converts convert the data in ```$GLUE_DIR``` into the correct format to train the typo corrector. This saves in ```
$TC_DIR/glue_tc_preprocessed
```. Next, cd to `scRNN`, and run:
```
python train.py --task_name mrpc --preprocessed_glue_dir $TC_DIR/glue_tc_preprocessed --tc_dir $TC_DIR
```
This trains a typo-corrector based on random perturbations to the MRPC data. The typo corrector is saved at `$TC_DIR/model_dumps` and the associated vocab (necessary) is saved at `TC_DIR/vocab` (both will likely have to be premade in codalab. Now, we can repeat the original run except with ```--recoverer scrnn``` and ```tc_dir $TC_DIR```.

### Connected Component Clusters.
Finally, we're done with the baselines! To try using clusters as a defense, we use:
```
python run_glue.py --task_name $TASK_NAME --do_lower_case --do_train --do_eval --data_dir $GLUE_DIR/$TASK_NAME --output_dir model_output/$TASK_NAME --overwrite_output_dir --seed_output_dir --save_results --save_dir codalab --recoverer clust-rep --clusterer_path $CLUSTERER_PATH --augmentor identity --run_test --do_robust
```
Here, we include ```clusterer_path``` to load the mapping, and ```do_robust``` to compute the actual robust accuracy.

### Agglomerative Clusters. 
We will now construct our more complicated clusters, the agglomerative clusters. To leverage existing connected components for computational constraints, we parellelize. To do so, first make the directory where the two partial clusteres will be stored: `$clusterers/vocab100000_ed1_gamma0.3$`. Once the directory is made, run, in parallel: 
```
python agglom_clusters.py --gamma 0.3 --clusterer_path $CLUSTERER_PATH --job_id 0 --num_jobs 2
python agglom_clusters.py --gamma 0.3 --clusterer_path $CLUSTERER_PATH --job_id 1 --num_jobs 2
```
This will save two partial clusterers. To combine them (after both jobs are complete) run:
```
python reconstruct_clusterers.py --clusterer_dir clusterers/vocab100000_ed1_gamma0.3
```
This will save the clusterer at ```clusterers/vocab100000_ed1_gamma0.3.pkl```. Finally, run the identical commands as connected component clusters, but first use ```export CLUSTERER_PATH=clusterers/vocab100000_ed1_gamma0.3.pkl``` to run. Other value of gamma (only needed for SST-2) are loaded from premade saved files (from exactly this process) in ```saved_clusterers```.

### Internal permutation experiments
Much of the code remains the same for internal permutations. Just use ```--perturb_type intprm``` when constructing the clusters, ```--attack_type intprm``` when using an internal permutation attack, and ```--recoverer clust-intprm``` to use an internal permutation recoverer.

