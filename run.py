import os 
import json
import os
import logging
import argparse
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import DataLoader

from models import TextEncoder, AudioEncoder, EarlyFusionEncoderFinetune, LateFusionEncoderFinetune
from data_loader import TextDataset, AudioDataset, MultimodalDataset
from train_validate import train_model, validate_model
from utils import set_seed, NUM_FOLDS, set_logger, get_weights


def run_model(opt):
    for key, value in vars(opt).items():
        if value in ["True", "False"]:
            value = value=="True"
        setattr(opt, key, value)

    dataset_df = pd.read_csv(opt.data_path, sep="\t")

    with open(opt.dataset_fold_path, "r") as f:
        data_folds = json.load(f)

    dataset_df = dataset_df.dropna()
    session_ids = list(set(dataset_df["session"].values.tolist()))
    session_ids.sort()

    output_dir = os.path.join(opt.out_path, opt.model, opt.output_filename)
    os.makedirs(output_dir, exist_ok=True)
    set_logger(os.path.join(output_dir, "logging.log"))

    for key, value in vars(opt).items():
        logging.info("{}, {}".format(key, value))

    all_results_df = pd.DataFrame([])

    for fold in range(NUM_FOLDS):
        opt.fold = fold
        log_str = "-"*15 + "split " + str(fold) + "-"*15
        logging.info(log_str)
        train_ids, val_ids, test_ids = data_folds[str(fold)]

        if opt.model == "early_fusion_finetune":
            model = EarlyFusionEncoderFinetune(opt)
            Dataset = MultimodalDataset
        elif opt.model == "late_fusion_finetune":
            model = LateFusionEncoderFinetune(opt)
            Dataset = MultimodalDataset
        elif opt.model == "audio":
            model = AudioEncoder(opt)
            Dataset = AudioDataset
        elif opt.model == "text":
            model = TextEncoder(opt)
            Dataset = TextDataset
        model = model.cuda()

        train_data = Dataset(train_ids, dataset_df, opt)
        val_data = Dataset(val_ids, dataset_df, opt)
        test_data = Dataset(test_ids, dataset_df, opt)

        train_labels = train_data.get_labels()
        val_labels = val_data.get_labels()
        test_labels = test_data.get_labels()

        print("train/val/test size", len(train_data), len(val_data), len(test_data))
        print("train/val/test session size", len(train_ids), len(val_ids), len(test_ids))
        print("Class Dist train:", Counter(train_labels), "val:", Counter(val_labels), "test:", Counter(test_labels))

        train_dl = DataLoader(train_data, num_workers=opt.num_worker, batch_size=opt.batch_size, collate_fn=train_data.collate_fn, shuffle=True, pin_memory=True, drop_last=True)
        val_dl = DataLoader(val_data, num_workers=opt.num_worker, batch_size=opt.batch_size, collate_fn=val_data.collate_fn, shuffle=False, pin_memory=True)
        test_dl = DataLoader(test_data, num_workers=opt.num_worker, batch_size=opt.batch_size, collate_fn=test_data.collate_fn, shuffle=False, pin_memory=True)

        weights = get_weights(train_labels) 
        print("class weights {:.2f}, {:.2f}".format(weights[1], weights[0]))
        if opt.model in ["late_fusion_finetune"]:
            loss_func = torch.nn.NLLLoss(weight=torch.tensor(weights).cuda())
            optimizer = torch.optim.AdamW([
                {"params": model.modality_ratio, "lr": 1e-2},
                {"params": model.audio_encoder.parameters()},
                {"params": model.text_encoder.parameters()}],
                lr=opt.learning_rate, weight_decay=opt.weight_decay)
        else:
            loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda())
            optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=opt.patience, verbose=True)

        last_best_epoch = -1
        best_score = -1 
        for epoch in range(1, opt.epochs_num+1):
            log_str = "Epoch: {}/{}".format(epoch, opt.epochs_num)
            logging.info(log_str)
            loss, train_score = train_model(model, train_dl, optimizer, loss_func, opt)
            val_loss, val_score, _ = validate_model(model, val_dl, loss_func, opt)
            scheduler.step(val_loss)

            if val_score > best_score:
                best_score = val_score
                last_best_epoch = epoch
                best_state = {"epoch":epoch, "model":model.state_dict(), "opt":opt}
                torch.save(best_state, os.path.join(output_dir, "{}_{}.pt".format(opt.model_name, fold)))

            log_str = "Train Loss: {:.4f} Train F1: {:.3f} Val F1: {:.3f}".format(loss, train_score, val_score)
            logging.info(log_str)

            if epoch - last_best_epoch >= opt.patience:
                # Early stopping
                break

        model.load_state_dict(torch.load(os.path.join(output_dir, "{}_{}.pt".format(opt.model_name, fold)))["model"])
        _, test_score, result_df = validate_model(model, test_dl, loss_func, opt)
        result_df["fold"] = [fold] * result_df.shape[0]
        all_results_df = pd.concat([all_results_df, result_df], axis=0)

        final_log_str = "Test F1 for fold {}: {:.3f}\n".format(fold, test_score) + "*"*15 + "\n"
        logging.info(final_log_str)
        if opt.model in ["late_fusion_finetune"]:
            log_str = f"Modality ratio: {model.sigmoid(model.modality_ratio)}\n"
            logging.info(log_str)

    all_results_df.set_index("ID", inplace=True)
    all_results_df.sort_index(inplace=True)
    all_results_df.to_csv(os.path.join(output_dir, "predictions.csv"), header=True, index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--data_path", type=str, default="./data/combined_dataset_feats.tsv")
    parser.add_argument("--dataset_fold_path", type=str, default="./data/independent_dataset_folds.json")
    parser.add_argument("--out_path", type=str, default="./exps_independent")
    parser.add_argument("--dataset", type=str, default="combined")
    parser.add_argument("--output_filename", type=str)
    parser.add_argument("--model_name", type=str)

    parser.add_argument("--by_speaker", type=str, default="both")
    parser.add_argument("--quartile", type=int, default=-1)

    # model
    parser.add_argument("--model", type=str, default="text")

    parser.add_argument("--context_window", type=int, default=20)
    parser.add_argument("--hop", type=int, default=10)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--class_num", type=int, default=2)

    # text model
    parser.add_argument("--text_model", type=str, default="distil-roberta-emotion")
    parser.add_argument("--finetune", type=int, default="2")
    parser.add_argument("--hidden_dim_1", type=int, default=128)
    parser.add_argument("--hidden_dim_2", type=int, default=128)
    parser.add_argument("--nlayer", type=int, default=2)

    parser.add_argument("--bidirectional", type=str, default="True")
    parser.add_argument("--encoder_pooling", type=str, default="cls")
    parser.add_argument("--self_attn", type=str, default="True")
    parser.add_argument("--attn_pooling", type=str, default="mean_max")
    parser.add_argument("--attn_heads", type=int, default=1)
    parser.add_argument("--speaker_encoding", type=str, default="True")

    # audio base model
    parser.add_argument("--audio_feat", type=str, default="hubert")
    parser.add_argument("--feat_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=128)

    # training
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs_num", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)

    opt = parser.parse_args()

    set_seed()
    run_model(opt)
