import torch 
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score


def train_model(model, train_dl, optimizer, loss_func, opt):
    all_labels, all_predictions = [], []
    total_train_loss = 0.0
    cnt = 0

    model.train()
    for batch in tqdm(train_dl):
        optimizer.zero_grad()

        if opt.model in ["early_fusion", "early_fusion_finetune", "late_fusion", "late_fusion_finetune", "x_norm", "x_norm_finetune"]:
            input_ids, attn_mask, lengths, audio_features, labels, seq_lengths, speakers, global_ids = batch
            input_ids = input_ids.cuda()
            attn_mask = attn_mask.cuda()
            audio_features = audio_features.cuda()
            labels = labels.cuda()
            speakers = speakers.cuda() if opt.speaker_encoding else None
            batch_features = [input_ids, attn_mask, lengths, audio_features, seq_lengths, speakers, global_ids]
            logits = model(batch_features)

        elif opt.model == "audio":
            features, lengths, labels, speakers, global_ids = batch

            features = features.cuda()
            labels = labels.cuda()
            speakers = speakers.cuda() if opt.speaker_encoding else None

            logits = model(features, lengths, speakers)

        elif opt.model == "text":
            input_ids, attn_mask, lengths, labels, speakers, _ = batch

            input_ids = input_ids.cuda()
            attn_mask = attn_mask.cuda()
            labels = labels.cuda()
            speakers = speakers.cuda() if opt.speaker_encoding else None

            logits = model(input_ids, attn_mask, lengths, speakers)

        loss = loss_func(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.gradient_clip)
        optimizer.step()

        with torch.no_grad():
            predictions = np.argmax(logits.detach().cpu(), axis=1).tolist()
            all_predictions += predictions
            
            labels = labels.detach().cpu().numpy()
            labels = labels.reshape(1,-1).tolist()[0]
            all_labels += labels
            total_train_loss += loss.item()
            cnt += 1

    score = f1_score(all_labels, all_predictions, average="macro")

    return total_train_loss/cnt, score


def validate_model(model, dl, loss_func, opt):
    all_labels, all_predictions, all_sess_ids = [], [], []
    total_loss = 0.0
    cnt = 0

    model.eval()
    with torch.no_grad(): 
        for batch in tqdm(dl):
            if opt.model in ["early_fusion", "early_fusion_finetune", "late_fusion", "late_fusion_finetune", "x_norm", "x_norm_finetune"]:
                input_ids, attn_mask, lengths, audio_features, labels, seq_lengths, speakers, global_ids = batch
                input_ids = input_ids.cuda()
                attn_mask = attn_mask.cuda()
                audio_features = audio_features.cuda()
                labels = labels.cuda()
                speakers = speakers.cuda() if opt.speaker_encoding else None
                batch_features = [input_ids, attn_mask, lengths, audio_features, seq_lengths, speakers, global_ids]
                logits = model(batch_features)

            elif opt.model == "audio":
                features, lengths, labels, speakers, global_ids = batch

                features = features.cuda()
                labels = labels.cuda()
                speakers = speakers.cuda() if opt.speaker_encoding else None

                logits = model(features, lengths, speakers)

            elif opt.model == "text":
                input_ids, attn_mask, lengths, labels, speakers, global_ids = batch

                input_ids = input_ids.cuda()
                attn_mask = attn_mask.cuda()
                labels = labels.cuda()
                speakers = speakers.cuda() if opt.speaker_encoding else None

                logits = model(input_ids, attn_mask, lengths, speakers)

            loss = loss_func(logits, labels)
            total_loss += loss.item()

            labels = labels.detach().cpu().numpy().tolist()
            predictions = np.argmax(logits.detach().cpu(), axis=1).tolist()

            all_labels += labels
            all_predictions += predictions
            all_sess_ids += global_ids
            cnt += 1

    score = f1_score(all_labels, all_predictions, average="macro")
    results_df = pd.DataFrame({"labels":all_labels, "predictions":all_predictions, "ID":all_sess_ids})

    return total_loss/cnt, score, results_df
