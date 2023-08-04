import torch
import numpy as np

from utils import model_dict
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, session_ids, dataset_df, opt):
        self.session_ids = session_ids 
        data_df = dataset_df[dataset_df["session"].isin(session_ids)]

        if opt.quantile != -1:
            data_df = data_df[data_df["quant"]==opt.quantile]

        if opt.by_speaker == "therapist":
            data_df = data_df[data_df["speaker"]=="I"]
        elif opt.by_speaker == "client":
            data_df = data_df[data_df["speaker"]=="P"]

        self.data_df = data_df
        self.tokenizer = AutoTokenizer.from_pretrained(model_dict[opt.text_model])
        self.input_ids, self.attn_mask, self.lengths, self.speakers, self.global_ids, self.labels = [], [], [], [], [], []

        for sess_id in self.session_ids: 
            sess_df = self.data_df[self.data_df["session"]==sess_id]
            sess_len = len(sess_df)
            sess_label = sess_df.empathy_label.values[0]
            _global_ids = sess_df["global_id"].tolist()
            _speakers = sess_df["speaker"].tolist()

            utts = sess_df["normed_text"].tolist()
            encodings = self.tokenizer(utts, truncation=True, padding=True)
            _input_ids = encodings["input_ids"]
            _attn_mask = encodings["attention_mask"]

            offsets = range(0, sess_len, opt.hop)
            for offset in offsets:
                start_idx = offset 
                end_idx = min(offset+opt.context_window, sess_len)

                global_ids = _global_ids[start_idx]

                input_ids_tensor = torch.LongTensor(_input_ids[start_idx:end_idx])
                attn_mask_tensor = torch.LongTensor(_attn_mask[start_idx:end_idx])
                speakers_tensor = torch.LongTensor([x == "P" for x in _speakers[start_idx:end_idx]])

                self.input_ids.append(input_ids_tensor)
                self.attn_mask.append(attn_mask_tensor)
                self.speakers.append(speakers_tensor)
                self.global_ids.append(global_ids)
                self.labels.append(sess_label)

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_mask[idx], self.labels[idx], self.speakers[idx], self.global_ids[idx]

    def collate_fn(self, data):
        input_ids, attn_mask, labels, speakers, global_ids = zip(*data)

        batch_size = len(input_ids)
        lengths_dim1 = [x.shape[0] for x in input_ids]
        lengths_dim2 = [x.shape[1] for x in input_ids]

        labels = torch.LongTensor(labels)
        global_ids = list(global_ids)

        input_ids_tensor = torch.zeros(batch_size, max(lengths_dim1), max(lengths_dim2)).long()
        attn_mask_tensor = torch.zeros(batch_size, max(lengths_dim1), max(lengths_dim2)).long()

        # initializing to 2 since the speaker codes are 0 and 1
        speaker_tensor = torch.ones(batch_size, max(lengths_dim1)).long() * 2

        for idx, (length1, length2) in enumerate(zip(lengths_dim1, lengths_dim2)):
            input_ids_tensor[idx, :length1, :length2] = input_ids[idx]
            attn_mask_tensor[idx, :length1, :length2] = attn_mask[idx]
            speaker_tensor[idx, :length1] = speakers[idx]

        return input_ids_tensor, attn_mask_tensor, lengths_dim1, labels, speaker_tensor, global_ids


class AudioDataset(Dataset):
    def __init__(self, session_ids, dataset_df, opt):
        self.data_root = opt.data_root
        self.session_ids = session_ids
        data_df = dataset_df[dataset_df["session"].isin(session_ids)]
        column_path = f"{opt.audio_feat}_path"

        if opt.quantile != -1:
            data_df = data_df[data_df["quant"]==opt.quantile]

        if opt.by_speaker == "therapist":
            data_df = data_df[data_df["speaker"]=="I"]
        elif opt.by_speaker == "client":
            data_df = data_df[data_df["speaker"]=="P"]

        self.data_df = data_df
        self.input_paths, self.speakers, self.global_ids, self.ids, self.labels = [], [], [], [], []

        for sess_id in self.session_ids: 
            sess_df = self.data_df[self.data_df["session"]==sess_id]
            sess_len = len(sess_df)
            sess_label = sess_df.empathy_label.values[0]
            _global_ids = sess_df["global_id"].tolist()
            _speakers = sess_df["speaker"].tolist()

            utt_paths = sess_df[column_path].tolist()

            offsets = range(0, sess_len, opt.hop)
            for offset in offsets:
                start_idx = offset 
                end_idx = min(offset+opt.context_window, sess_len)

                paths = utt_paths[start_idx:end_idx]
                speakers_tensor = torch.LongTensor([x == "P" for x in _speakers[start_idx:end_idx]])
                global_ids = _global_ids[start_idx]

                self.input_paths.append(paths)
                self.speakers.append(speakers_tensor)
                self.global_ids.append(global_ids)
                self.labels.append(sess_label)

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        return self.input_paths[idx], self.labels[idx], self.speakers[idx], self.global_ids[idx],

    def collate_fn(self, data):
        input_paths, labels, speakers, global_ids = zip(*data)

        labels = torch.LongTensor(labels)
        global_ids = list(global_ids)

        seq_frames = []
        for seq in input_paths:
            utt_frames = []
            for utt_file in seq:
                feat = np.load(utt_file.replace("/data/perception-temp/ttran", self.data_root)) # [T, feat_dim]
                mean_frames = np.mean(feat, 0)
                max_frames = np.max(feat, 0)
                mean_max = np.hstack([mean_frames, max_frames])
                utt_frames.append(mean_max)
            utt_frames = np.stack(utt_frames, axis=0)
            utt_frames = torch.from_numpy(utt_frames)
            seq_frames.append(utt_frames)

        batch_size = len(data)
        lengths_dim = [len(x) for x in input_paths]
        feat_dim = utt_frames.shape[1]
        feature_tensor = torch.zeros(batch_size, max(lengths_dim), feat_dim)
        speaker_tensor = torch.ones(batch_size, max(lengths_dim)).long()*2 # initializing to 2 since the speaker codes are 0 and 1
        for idx, (length, utt_frames) in enumerate(zip(lengths_dim, seq_frames)):
            feature_tensor[idx, :length, :] = utt_frames
            speaker_tensor[idx, :length] = speakers[idx]

        return feature_tensor, lengths_dim, labels, speaker_tensor, global_ids


class MultimodalDataset(Dataset):
    def __init__(self, session_ids, dataset_df, opt):
        self.data_root = opt.data_root
        self.session_ids = session_ids 
        data_df = dataset_df[dataset_df["session"].isin(session_ids)]
        column_path = f"{opt.audio_feat}_path"

        if opt.quantile != -1:
            data_df = data_df[data_df["quant"]==opt.quantile]

        if opt.by_speaker == "therapist":
            data_df = data_df[data_df["speaker"]=="I"]
        elif opt.by_speaker == "client":
            data_df = data_df[data_df["speaker"]=="P"]

        self.data_df = data_df
        self.tokenizer = AutoTokenizer.from_pretrained(model_dict[opt.text_model])
        self.input_ids, self.attn_mask, self.lengths, self.speakers, self.global_ids, self.labels, self.input_paths = [], [], [], [], [], [], []

        for sess_id in self.session_ids: 
            sess_df = self.data_df[self.data_df["session"]==sess_id]
            sess_len = len(sess_df)
            sess_label = sess_df.empathy_label.values[0]
            _global_ids = sess_df["global_id"].tolist()
            _speakers = sess_df["speaker"].tolist()

            utts = sess_df["normed_text"].tolist()
            encodings = self.tokenizer(utts, truncation=True, padding=True)
            _input_ids = encodings["input_ids"]
            _attn_mask = encodings["attention_mask"]
            utt_paths = sess_df[column_path].tolist()

            offsets = range(0, sess_len, opt.hop)
            for offset in offsets:
                start_idx = offset 
                end_idx = min(offset+opt.context_window, sess_len)

                global_ids = _global_ids[start_idx]
                input_ids_tensor = torch.LongTensor(_input_ids[start_idx:end_idx])
                attn_mask_tensor = torch.LongTensor(_attn_mask[start_idx:end_idx])
                speakers_tensor = torch.LongTensor([x == "P" for x in _speakers[start_idx:end_idx]])
                paths = utt_paths[start_idx:end_idx]

                self.input_ids.append(input_ids_tensor)
                self.attn_mask.append(attn_mask_tensor)
                self.speakers.append(speakers_tensor)
                self.global_ids.append(global_ids)
                self.labels.append(sess_label)
                self.input_paths.append(paths)

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_mask[idx], self.labels[idx], self.speakers[idx], self.global_ids[idx], self.input_paths[idx]

    def collate_fn(self, data):
        input_ids, attn_mask, labels, speakers, global_ids, input_paths = zip(*data)

        batch_size = len(input_ids)
        lengths_dim1 = [x.shape[0] for x in input_ids]
        lengths_dim2 = [x.shape[1] for x in input_ids]

        labels = torch.LongTensor(labels)
        global_ids = list(global_ids)

        input_ids_tensor = torch.zeros(batch_size, max(lengths_dim1), max(lengths_dim2)).long()
        attn_mask_tensor = torch.zeros(batch_size, max(lengths_dim1), max(lengths_dim2)).long()
        speaker_tensor = torch.ones(batch_size, max(lengths_dim1)).long()*2 # initializing to 2 since the speaker codes are 0 and 1

        for idx, (length1, length2) in enumerate(zip(lengths_dim1, lengths_dim2)):
            input_ids_tensor[idx, :length1, :length2] = input_ids[idx]
            attn_mask_tensor[idx, :length1, :length2] = attn_mask[idx]
            speaker_tensor[idx, :length1] = speakers[idx]

        seq_frames = []
        seq_lengths = []
        for seq in input_paths:
            utt_frames = []
            frame_lengths = [] 
            for utt_file in seq:
                feat = np.load(utt_file.replace("/data/perception-temp/ttran", self.data_root)) # [T, feat_dim]
                frame_lengths.append(feat.shape[0])
                mean_frames = np.mean(feat, 0)
                max_frames = np.max(feat, 0)
                mean_max = np.hstack([mean_frames, max_frames])
                utt_frames.append(mean_max)
            seq_lengths.append(frame_lengths)
            utt_frames = np.stack(utt_frames, axis=0)
            utt_frames = torch.from_numpy(utt_frames)
            seq_frames.append(utt_frames)

        feat_dim = utt_frames.size(1)
        feature_tensor = torch.zeros(batch_size, max(lengths_dim1), feat_dim)
        for idx, (length, utt_frames) in enumerate(zip(lengths_dim1, seq_frames)):
            feature_tensor[idx, :length, :] = utt_frames

        return input_ids_tensor, attn_mask_tensor, lengths_dim1, feature_tensor, labels, seq_lengths, speaker_tensor, global_ids
