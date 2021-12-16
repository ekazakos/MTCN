import torch
from torch import nn
from embeddings import FeatureEmbedding
from transformers import TransformerEncoder, TransformerEncoderLayer


class MTCN_AV(nn.Module):
    def __init__(self,
                 num_class,
                 seq_len=5,
                 num_clips=10,
                 visual_input_dim=2304,
                 audio_input_dim=2304,
                 d_model=512,
                 dim_feedforward=2048,
                 nhead=8,
                 num_layers=6,
                 dropout=0.1,
                 classification_mode='summary',
                 audio=True):
        super(MTCN_AV, self).__init__()
        self.num_class = num_class
        self.seq_len = seq_len
        self.num_clips = num_clips
        self.visual_input_dim = visual_input_dim
        self.audio_input_dim = audio_input_dim
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        print("Building Transformer with {}-D, {} heads, and {} layers".format(self.d_model,
                                                                               self.nhead,
                                                                               self.num_layers))
        assert classification_mode in ['all', 'summary'], \
            "Classification mode not supported. Choose from ['all', 'summary']"
        self.classification_mode = classification_mode
        print("Classification mode: {}".format(self.classification_mode))
        self.audio = audio
        self._create_model()

    def _create_model(self):
        self.feature_embedding = FeatureEmbedding(self.seq_len,
                                                  self.num_clips,
                                                  self.visual_input_dim,
                                                  self.audio_input_dim,
                                                  self.d_model,
                                                  self.audio,
                                                  not isinstance(self.num_class, (list, tuple)))
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model,
                                                nhead=self.nhead,
                                                dim_feedforward=self.dim_feedforward,
                                                dropout=self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        if isinstance(self.num_class, (list, tuple)):
            self.fc_verb = nn.Linear(self.d_model, self.num_class[0])
            self.fc_noun = nn.Linear(self.d_model, self.num_class[1])
        else:
            self.fc_action = nn.Linear(self.d_model, self.num_class)

    def forward(self, inputs, extract_attn_weights=False):
        # Project audio and visual features to lower dim and add positional, modality, and summary embeddings
        x = self.feature_embedding(inputs)
        if extract_attn_weights:
            x, attn_weights = self.transformer_encoder(x)
            x = x.transpose(0, 1).contiguous()
        else:
            x, _ = self.transformer_encoder(x)
            x = x.transpose(0, 1).contiguous()
        if isinstance(self.num_class, (list, tuple)):
            if self.classification_mode == 'all':
                output_verb_av = self.fc_verb(x[:, :-2, :]).transpose(1, 2).contiguous()
                output_noun_av = self.fc_noun(x[:, :-2, :]).transpose(1, 2).contiguous()
                output_verb_ve = self.fc_verb(x[:, -2, :]).unsqueeze(2)
                output_noun_no = self.fc_noun(x[:, -1, :]).unsqueeze(2)
                output_verb = torch.cat([output_verb_av, output_verb_ve], dim=2)
                output_noun = torch.cat([output_noun_av, output_noun_no], dim=2)
            else:
                output_verb = self.fc_verb(x[:, -2, :])
                output_noun = self.fc_noun(x[:, -1, :])
            output = (output_verb, output_noun)
        else:
            if self.classification_mode == 'all':
                output = self.fc_action(x).transpose(1, 2).contiguous()
            else:
                output = self.fc_action(x[:, -1, :])
        if extract_attn_weights:
            return output, attn_weights
        else:
            return output
