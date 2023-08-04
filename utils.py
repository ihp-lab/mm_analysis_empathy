import os
import torch
import random
import logging
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score


#https://huggingface.co/j-hartmann/emotion-english-distilroberta-base (emo-distil-roberta)
#https://huggingface.co/arpanghoshal/EmoRoBERTa (emo-roberta)
model_dict = {
    "bert": "bert-base-uncased", 
    "roberta":"roberta-base", 
    "distilbert":"distilbert-base-uncased",
    "distil-roberta": "distilroberta-base",
    "emotion":"bhadresh-savani/bert-base-uncased-emotion", 
    "distil-emotion":"bhadresh-savani/distilbert-base-uncased-emotion", 
    "distil-roberta-emotion": "j-hartmann/emotion-english-distilroberta-base", 
    "emo-roberta":"arpanghoshal/EmoRoBERTa", 
    "psych": "mnaylor/psychbert-cased", 
    "mental":"mental/mental-bert-base-uncased", 
}
NUM_FOLDS = 5
#NUM_FOLDS = 1


def set_seed():
    seed = 100
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def get_weights(labels):
    class_num = len(set(labels))
    weights = []
    for c in range(class_num):
        if labels.count(c) > 0:
            weights.append(len(labels)/labels.count(c))
        else:
            weights.append(0)

    if sum(weights) > 0:
        weights = [float(i) / sum(weights) for i in weights]
    else:
        weights = [1.0]*len(weights)
    
    return weights


class Object(object):
    pass
