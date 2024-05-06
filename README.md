# Charmer: Revisiting character-level adversarial attacks

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)[![Licence](https://img.shields.io/badge/MIT_License-lightgreen?style=for-the-badge)](./LICENSE)[![OpenReview](https://img.shields.io/badge/OpenReview-8C1B13?style=for-the-badge)](https://openreview.net/forum?id=AZWqXfM6z9)

![](schematic.png)

Official implementation in PyTorch of the **ICML'24 paper "Revisiting character-level adversarial attacks"**.

## Requirements
```
conda create -n lmrobust python=3.8
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c anaconda jupyter
pip install -r requirements.txt
```
`baseline` includes auxiliary code not written by authors.

## Sample scripts

To run Charmer in the TextAattack BERT, RoBERTa and ALBERT models, run the separate dataset scripts in `scripts`, e.g.:
```
cd scripts
bash run_sst.sh
```
This will produce a `.csv` file similar to:
```
original,perturbed,True,Pred_original,Pred_perturbed,success,Dist_char,Dist_token,similarity,time
Fears for T N pension after talks Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.,Fears for T E pension :fter talks Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.,2,2,3,True,2,4,0.9770777225494384,3.747948408126831
```

where each column is:
- `original`: Original sentence.
- `perturbed`: Sentence after the attack.
- `True`: True label of the sentence.
- `Pred_original`: Prediction of the model for the original sentence.
- `Pred_perturbed`: Prediction of the model for the attacked sentence.
- `success`: True if `Pred_original` != `Pred_perturbed`.
- `Dist_char`: Levenshtein distance in the character-level between `original` and `perturbed`.
- `Dist_token`: Levenshtein distance in the token-level between `original` and `perturbed`.
- `similarity`: USE between `original` and `perturbed`.
- `time`: Time needed to produce `perturbed`.

## Cite as:

```
@inproceedings{Abad2024Charmer,
  author = {Abad Rocamora, Elias and Wu, Yongtao and Liu, Fanghui and Chrysos, Grigorios G and Cevher, Volkan},

  title = {Revisiting character-level adversarial attacks},

  booktitle = {International Conference on Machine Learning (ICML)},

  year = {2024}
}
```