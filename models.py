import torch
import torch.nn as nn

from utils import model_dict
from transformers import AutoModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextEncoder(nn.Module):
    def __init__(self, opt):
        super(TextEncoder, self).__init__()
 
        self.bert = AutoModel.from_pretrained(model_dict[opt.text_model])
        input_dim = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False

        if opt.finetune >= 1:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for param in self.bert.encoder.layer[-1].parameters():
                param.requires_grad = True

            if opt.finetune >= 2:
                for param in self.bert.encoder.layer[-2].parameters():
                    param.requires_grad = True

        self.pooling_type = opt.encoder_pooling
        self.self_attn = opt.self_attn
        self.speaker_encoding = opt.speaker_encoding

        self.downsize_linear = nn.Linear(input_dim if not opt.speaker_encoding else input_dim*2, opt.hidden_dim_1)
        self.GRU = nn.GRU(opt.hidden_dim_1, opt.hidden_dim_2, num_layers=opt.nlayer, bidirectional=opt.bidirectional, batch_first=True)

        classifier_hidden_dim = opt.hidden_dim_2*2 if opt.bidirectional else opt.hidden_dim_2
        classifier_hidden_dim = classifier_hidden_dim*2 if (self.self_attn and opt.attn_pooling == 'mean_max') else classifier_hidden_dim

        self.linear = nn.Linear(classifier_hidden_dim, opt.class_num)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=opt.dropout)

        if self.self_attn:
            self.attn_pooling = opt.attn_pooling
            self.attnfunc = nn.MultiheadAttention(opt.hidden_dim_2*2 if opt.bidirectional else opt.hidden_dim_2, num_heads=opt.attn_heads, batch_first=True)

        if self.speaker_encoding: 
            self.speaker_emb = nn.Embedding(3, input_dim)

        self.classifier = nn.Sequential(
                            self.dropout,
                            self.linear)

    def extract_features(self, input_ids, attn_mask, lengths, speakers): 
        batch_size, _, token_count = input_ids.shape        
        input_ids = input_ids.reshape(-1, token_count)
        attn_mask = attn_mask.reshape(-1, token_count)

        embeddings = self.bert(input_ids, attention_mask=attn_mask, output_hidden_states=False)

        if self.pooling_type == 'mean':
            embeddings = embeddings.pooler_output
        elif self.pooling_type == 'cls':
            embeddings = embeddings.last_hidden_state
            embeddings = embeddings[:,0,:]

        if self.speaker_encoding: 
            speaker_embeddings = self.speaker_emb(speakers)
            speaker_embeddings = speaker_embeddings.reshape(embeddings.shape)
            embeddings = torch.cat((embeddings, speaker_embeddings), 1)

        embeddings = self.downsize_linear(embeddings)

        nhid = embeddings.shape[-1]
        embeddings = embeddings.reshape(batch_size, -1, nhid)

        embeddings_packed = pack_padded_sequence(embeddings, lengths, enforce_sorted=False, batch_first=True)
        self.GRU.flatten_parameters()

        outputs, _ = self.GRU(embeddings_packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        if self.self_attn: 
            outputs, _ = self.attnfunc(outputs, outputs, outputs)
            if self.attn_pooling == 'mean':
                outputs = torch.mean(outputs, dim=1)
            elif self.attn_pooling == 'max':
                outputs, _ = torch.max(outputs, dim=1)
            elif self.attn_pooling == 'mean_max': 
                outputs_max, _ = torch.max(outputs, dim=1)
                outputs_mean = torch.mean(outputs, dim=1)
                outputs = torch.cat((outputs_mean, outputs_max), dim=1)
        else:
            outputs = outputs[:,-1,:]

        return outputs

    def forward(self, input_ids, attn_mask, lengths, speakers):
        features = self.extract_features(input_ids, attn_mask, lengths, speakers)
        outputs = self.classifier(features)

        return outputs


class AudioEncoder(nn.Module):
    def __init__(self, opt):
        super(AudioEncoder, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.feat_dim = opt.feat_dim * 2 # audio features pre-pooled with mean&max
        self.speaker_encoding = opt.speaker_encoding

        self.proj = nn.Linear(self.feat_dim, self.hidden_dim)
        input_dim = self.hidden_dim
        
        if self.speaker_encoding: 
            self.speaker_emb = nn.Embedding(3, opt.hidden_dim)
            input_dim += opt.hidden_dim

        self.audio_encoder = nn.GRU(input_size=input_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=2,
                                    dropout=opt.dropout,
                                    batch_first=True,
                                    bidirectional=True)
        
        self.classifier = nn.Sequential(
                    nn.Linear(self.hidden_dim*2, self.hidden_dim), # x2 for bidirectional
                    nn.ReLU(),
                    nn.BatchNorm1d(num_features=self.hidden_dim),
                    nn.Dropout(p=opt.dropout),
                    nn.Linear(self.hidden_dim, opt.class_num))
        
    def extract_features(self, features, speakers): 
        batch_size, utt_count, feat_dim = features.shape

        features = features.reshape(batch_size*utt_count, feat_dim)
        features = self.proj(features).reshape(batch_size, utt_count, self.hidden_dim)
        if self.speaker_encoding: 
            speaker_embeddings = self.speaker_emb(speakers)
            features = torch.cat([features, speaker_embeddings], dim=2)

        features, _ = self.audio_encoder(features)
        features = features[:,-1,:]

        return features

    def forward(self, features, lengths, speakers):
        features = self.extract_features(features, speakers)
        outputs = self.classifier(features)

        return outputs


class EarlyFusionEncoder(nn.Module):
    def __init__(self, opt):
        super(EarlyFusionEncoder, self).__init__()

        self.audio_encoder = AudioEncoder(opt)
        self.text_encoder = TextEncoder(opt)

        self.text_out_dim = opt.hidden_dim_1 * 4
        self.audio_out_dim = opt.hidden_dim * 2
        self.hidden_dim = opt.hidden_dim
        self.classifier = nn.Sequential(
                    nn.Linear(self.text_out_dim+self.audio_out_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_features=self.hidden_dim),
                    nn.Dropout(p=opt.dropout),
                    nn.Linear(self.hidden_dim, opt.class_num))

    def forward(self, data):
        input_ids, attn_mask, lengths, audio_features, _, speakers, _ = data

        text_embeddings = self.text_encoder.extract_features(input_ids, attn_mask, lengths, speakers)
        audio_embeddings = self.audio_encoder.extract_features(audio_features)

        embeddings = torch.cat([text_embeddings, audio_embeddings], dim=1)
        fused_out = self.classifier(embeddings)

        return fused_out


class EarlyFusionEncoderFinetune(nn.Module):
    def __init__(self, opt):
        super(EarlyFusionEncoderFinetune, self).__init__()

        self.audio_encoder = AudioEncoder(opt)
        self.text_encoder = TextEncoder(opt)
        if opt.quantile == 3:
            quantile = 1
        elif opt.quantile == -1:
            quantile = 2
        elif opt.quantile == 1:
            quantile = 3

        self.audio_encoder.load_state_dict(torch.load("./{}/combined/audio/{}-{}/audio_{}.pt".format(opt.out_path, opt.by_speaker, quantile, opt.fold))["model"])
        self.text_encoder.load_state_dict(torch.load("./{}/combined/text/{}-{}/text_{}.pt".format(opt.out_path, opt.by_speaker, quantile, opt.fold))["model"])

        self.text_out_dim = opt.hidden_dim_1 * 4
        self.audio_out_dim = opt.hidden_dim * 2
        self.hidden_dim = opt.hidden_dim
        self.classifier = nn.Sequential(
                    nn.Linear(self.text_out_dim+self.audio_out_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_features=self.hidden_dim),
                    nn.Dropout(p=opt.dropout),
                    nn.Linear(self.hidden_dim, opt.class_num))

    def forward(self, data):
        input_ids, attn_mask, lengths, audio_features, _, speakers, _ = data

        text_embeddings = self.text_encoder.extract_features(input_ids, attn_mask, lengths, speakers)
        audio_embeddings = self.audio_encoder.extract_features(audio_features)

        embeddings = torch.cat([text_embeddings, audio_embeddings], dim=1)
        fused_out = self.classifier(embeddings)

        return fused_out


class LateFusionEncoder(nn.Module):
    def __init__(self, opt):
        super(LateFusionEncoder, self).__init__()

        self.audio_encoder = AudioEncoder(opt)
        self.text_encoder = TextEncoder(opt)
        self.modality_ratio = nn.Parameter(torch.randn(1))
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        input_ids, attn_mask, lengths, audio_features, seq_lengths, speakers, _ = data

        audio_logits = self.audio_encoder(audio_features, seq_lengths, speakers)
        text_logits = self.text_encoder(input_ids, attn_mask, lengths, speakers)
        fused_out = self.sigmoid(self.modality_ratio) * self.softmax(audio_logits) + (1-self.sigmoid(self.modality_ratio)) * self.softmax(text_logits)

        return fused_out


class LateFusionEncoderFinetune(nn.Module):
    def __init__(self, opt):
        super(LateFusionEncoderFinetune, self).__init__()

        self.audio_encoder = AudioEncoder(opt)
        self.text_encoder = TextEncoder(opt)
        if opt.quantile == 3:
            quantile = 1
        elif opt.quantile == -1:
            quantile = 2
        elif opt.quantile == 1:
            quantile = 3

        self.audio_encoder.load_state_dict(torch.load("./{}/combined/audio/{}-{}/audio_{}.pt".format(opt.out_path, opt.by_speaker, quantile, opt.fold))["model"])
        self.text_encoder.load_state_dict(torch.load("./{}/combined/text/{}-{}/text_{}.pt".format(opt.out_path, opt.by_speaker, quantile, opt.fold))["model"])
        self.modality_ratio = nn.Parameter(torch.randn(1))
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        input_ids, attn_mask, lengths, audio_features, seq_lengths, speakers, _ = data

        audio_logits = self.audio_encoder(audio_features, seq_lengths, speakers)
        text_logits = self.text_encoder(input_ids, attn_mask, lengths, speakers)
        fused_out = self.sigmoid(self.modality_ratio) * self.softmax(audio_logits) + (1-self.sigmoid(self.modality_ratio)) * self.softmax(text_logits)

        return fused_out


class XNorm(nn.Module):
    def __init__(self, opt):
        super(XNorm, self).__init__()

        self.audio_encoder = AudioEncoder(opt)
        self.text_encoder = TextEncoder(opt)
        self.modality_ratio = nn.Parameter(torch.randn(1))
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.text_out_dim = opt.hidden_dim_1 * 4
        self.audio_out_dim = opt.hidden_dim * 2
        self.a2t = nn.Linear(self.audio_out_dim, self.text_out_dim*2)
        self.t2a = nn.Linear(self.text_out_dim, self.audio_out_dim*2)

    def forward(self, data):
        input_ids, attn_mask, lengths, audio_features, seq_lengths, speakers, _ = data

        x1 = self.text_encoder.extract_features(input_ids, attn_mask, lengths, speakers)
        x2 = self.audio_encoder.extract_features(audio_features)

        a1, b1 = torch.mean(x1, [1], True), torch.std(x1, [1], keepdim=True).add(1e-8)
        a1, b1 = a1.repeat(1, x1.size(1)), b1.repeat(1, x1.size(1))

        a2, b2 = torch.chunk(self.a2t(x2), 2, dim=1)
        final_x1 = x1+((x1-a1)/b1)*b2+a2

        a2, b2 = torch.mean(x2, [1], True), torch.std(x2, [1], keepdim=True).add(1e-8)
        a2, b2 = a2.repeat(1, x2.size(1)), b2.repeat(1, x2.size(1))

        a1, b1 = torch.chunk(self.t2a(x1), 2, dim=1)
        final_x2 = x2+((x2-a2)/b2)*b1+a1

        text_logits = self.text_encoder.classifier(final_x1)
        audio_logits = self.audio_encoder.classifier(final_x2)

        fused_out = self.sigmoid(self.modality_ratio) * self.softmax(audio_logits) + (1-self.sigmoid(self.modality_ratio)) * self.softmax(text_logits)

        return fused_out


class XNormFineTune(nn.Module):
    def __init__(self, opt):
        super(XNormFineTune, self).__init__()

        self.audio_encoder = AudioEncoder(opt)
        self.text_encoder = TextEncoder(opt)
        if opt.quantile == 3:
            quantile = 1
        elif opt.quantile == -1:
            quantile = 2
        elif opt.quantile == 1:
            quantile = 3

        self.audio_encoder.load_state_dict(torch.load("./{}/combined/audio/{}-{}/audio_{}.pt".format(opt.out_path, opt.by_speaker, quantile, opt.fold))["model"])
        self.text_encoder.load_state_dict(torch.load("./{}/combined/text/{}-{}/text_{}.pt".format(opt.out_path, opt.by_speaker, quantile, opt.fold))["model"])
        self.modality_ratio = nn.Parameter(torch.randn(1))
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.text_out_dim = opt.hidden_dim_1 * 4
        self.audio_out_dim = opt.hidden_dim * 2
        self.a2t = nn.Linear(self.audio_out_dim, self.text_out_dim*2)
        self.t2a = nn.Linear(self.text_out_dim, self.audio_out_dim*2)

    def forward(self, data):
        input_ids, attn_mask, lengths, audio_features, seq_lengths, speakers, _ = data

        x1 = self.text_encoder.extract_features(input_ids, attn_mask, lengths, speakers)
        x2 = self.audio_encoder.extract_features(audio_features)

        a1, b1 = torch.mean(x1, [1], True), torch.std(x1, [1], keepdim=True).add(1e-8)
        a1, b1 = a1.repeat(1, x1.size(1)), b1.repeat(1, x1.size(1))

        a2, b2 = torch.chunk(self.a2t(x2), 2, dim=1)
        final_x1 = x1+((x1-a1)/b1)*b2+a2

        a2, b2 = torch.mean(x2, [1], True), torch.std(x2, [1], keepdim=True).add(1e-8)
        a2, b2 = a2.repeat(1, x2.size(1)), b2.repeat(1, x2.size(1))

        a1, b1 = torch.chunk(self.t2a(x1), 2, dim=1)
        final_x2 = x2+((x2-a2)/b2)*b1+a1

        text_logits = self.text_encoder.classifier(final_x1)
        audio_logits = self.audio_encoder.classifier(final_x2)

        fused_out = self.sigmoid(self.modality_ratio) * self.softmax(audio_logits) + (1-self.sigmoid(self.modality_ratio)) * self.softmax(text_logits)

        return fused_out
