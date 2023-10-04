<div align="center">
  <h1 align="center">Multimodal Analysis and Assessment of Therapist Empathy in Motivational Interviews</h1>
  <p align="center">

<a href="https://ttmt001.github.io/">
    Trang Tran</a>,
<a href="https://yufengyin.github.io/">
    Yufeng Yin</a>,
<a href="https://www.linkedin.com/in/leili-tavabi-92649693/">
    Leili Tavabi</a>,
<a href="https://profiles.ucsf.edu/joannalyn.delacruz">
    Joannalyn Delacruz</a>,
<a href="https://addictionresearch.ucsf.edu/people/brian-borsari-phd">
    Brian Borsari</a>,
<a href="https://woolleylab.ucsf.edu/principal-investigator">
    Joshua Woolley</a>,
<br>
<a href="https://schererstefan.net/">
    Stefan Scherer</a>,
<a href="https://people.ict.usc.edu/~soleymani/">
    Mohammad Soleymani</a>
<br>
<a href="https://ict.usc.edu/">USC ICT</a>, San Francisco VAHCS, UCSF Psychiatry and Behavioral Sciences

<strong>ICMI 2023</strong>
</p>
</div>

## Introduction

This is the official implementation of our ICMI 2023 paper: Multimodal Analysis and Assessment of Therapist Empathy in Motivational Interviews.

TODO

<p align="center">
  <img src="https://github.com/ihp-lab/mm_analysis_empathy/blob/main/pipeline.png" width="700px" />
</p>

## Installation
Clone repo:
```
git clone https://github.com/ihp-lab/mm_analysis_empathy.git
cd mm_analysis_empathy
```

The code is tested with Python == 3.10, PyTorch == 1.11.0 and CUDA == 11.3 on NVIDIA Quadro RTX 8000. We recommend you to use [anaconda](https://www.anaconda.com/) to manage dependencies.

```
conda create -n mm_empathy python=3.10
conda activate mm_empathy
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install pandas
pip install -U scikit-learn
pip install transformers==4.28.1
```

## Data
Sample data can be downloaded from Google Drive: [sample data](https://drive.google.com/file/d/1PeMqm-2xohMnnlUr4M-THWi94765lot1/view?usp=drive_link). Since the clinical data is not public, we have sampled a subset in addition to replacing a random set of transcripts with random tokens. The cross-validation folds are also for sample/toy use only.

Download and untar the file, then put `sample_data` under `./mm_analysis_empathy`.

## Checkpoints
Checkpoints are available on Google Drive: [exps_independent](https://drive.google.com/drive/folders/1hl5dswV38bRv-cG3CtGg5OKHTC7eGx0m?usp=drive_link) and [exps_dependent](https://drive.google.com/drive/folders/1iiFEnqAS4Tm69ANdoG-ew5EDIj1HMS0R?usp=drive_link).

Put `./exps_independent` and `./exps_dependent` under `./mm_analysis_empathy`.

## Training and Evaluation
### General notes:
* The json file `./sample_data/sample_data_folds.json` should be your cross-validation folds; supply the appropriate file depending on the therapist-independent vs. therapist-dependent settings.
* Quartiles ara 0-indexed, i.e. to train/evaluate on Q2 (like in our paper), set argument `--quartile 1`. `quartile=-1` uses the full sessions.
* The code assumes acoustic features are stored under `./sample_data/sample_features` 

```
CUDA_VISIBLE_DEVICES=0 python run.py --model {text,audio,early_fusion_finetune,late_fusion_finetune} 
    --model_name {text,audio,early_fusion_finetune,late_fusion_finetune} 
    --by_speaker {therapist,both} 
    --quartile {0,1,2,3,-1} 
    --output_filename {therapist,both}-quartile-{0,1,2,3,all} 
    --epochs_num 5
    --out_path ./exps_{dependent,independent}
    --data_root ./sample_data
    --data_path ./sample_data/sample_data_feats.tsv
    --dataset_fold_path ./sample_data/sample_data_folds.json 
```

For example, this is the command to train and evaluate on the unimodal text model, using only the therapist turns from Q2, assuming `sample_data_folds.json` contains cross-validation folds for the therapist-dependent setting:
```
CUDA_VISIBLE_DEVICES=0 python run.py --model text --model_name text 
    --by_speaker therapist 
    --quartile 1 
    --output_filename therapist-quartile-1 
    --epochs_num 5
    --out_path ./exps_dependent
    --data_root ./sample_data
    --data_path ./sample_data/sample_data_feats.tsv
    --dataset_fold_path ./sample_data/sample_data_folds.json 
```

Get overall f1 scores across folds
```
python compute_overall_scores.py --results PATH_TO_CSV_FILE
```
csv files are saved under `./OUT_PATH/MODEL_NAME/OUTPUT_FILENAME`.

## Citation
TODO

## Contact
If you have any questions, please raise an issue or email to Trang Tran (`ttran@ict.usc.edu`).

## Credits
Our codes are based on the following repositories.

- [text-empathy-recognition](https://github.com/ihp-lab/empathy-recognition-acii-2023)
- [multimodal-fusion](https://github.com/ihp-lab/XNorm)

