##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
from torch import nn
import numpy as np
from human_body_prior.models.vposer_model import NormalDistDecoder, VPoser
from transformers import AutoModel, AutoTokenizer

# Pose retrieval

class PoseText(nn.Module):
    def __init__(self, num_neurons=512, num_neurons_mini=32, latentD=512,
                 text_encoder_name='distilbertUncased', transformer_topping=None):
        super(PoseText, self).__init__()

        self.latentD = latentD

        # Define pose encoder
        self.pose_encoder = PoseEncoder(num_neurons, num_neurons_mini, latentD=latentD, role="retrieval")

        # Define text encoder
        self.text_encoder_name = text_encoder_name
        self.text_encoder = TransformerTextEncoder(self.text_encoder_name, latentD=latentD, topping=transformer_topping, role="retrieval")

        # Loss temperature
        self.loss_weight = torch.nn.Parameter( torch.FloatTensor((10,)) )
        self.loss_weight.requires_grad = True

    def forward(self, pose, captions, caption_lengths):
        pose_embs = self.pose_encoder(pose)
        text_embs = self.text_encoder(captions, caption_lengths)
        return pose_embs, text_embs

    def encode_raw_text(self, raw_text):
        self.tokenizer = TransformTokenizer()
        tokens = self.tokenizer(raw_text).to(device=self.loss_weight.device)
        length = torch.tensor([ len(tokens) ], dtype=tokens.dtype)
        text_embs = self.text_encoder(tokens.view(1, -1), length)
        return text_embs
        
    def encode_pose(self, pose):
        return self.pose_encoder(pose)

    def encode_text(self, captions, caption_lengths):
        return self.text_encoder(captions, caption_lengths)

class PoseEncoder(nn.Module):

    def __init__(self, num_neurons=512, num_neurons_mini=32, latentD=512, role=None):
        super().__init__()
        NB_INPUT_JOINTS = 52
        self.input_dim = NB_INPUT_JOINTS * 3

        # use VPoser pose encoder architecture...
        vposer_params = Object()
        vposer_params.model_params = Object() # type: ignore
        vposer_params.model_params.num_neurons = num_neurons # type: ignore
        vposer_params.model_params.latentD = latentD # type: ignore
        vposer = VPoser(vposer_params)
        encoder_layers = list(vposer.encoder_net.children())
        # change first layers to have the right data input size
        encoder_layers[1] = nn.BatchNorm1d(self.input_dim)
        encoder_layers[2] = nn.Linear(self.input_dim, num_neurons)
        # remove last layer; the last layer.s depend on the task/role
        encoder_layers = encoder_layers[:-1]
        
        # output layers
        if role == "retrieval":
            encoder_layers += [
                nn.Linear(num_neurons, num_neurons_mini), # keep the bottleneck while adapting to the joint embedding size
                nn.ReLU(),
                nn.Linear(num_neurons_mini, latentD),
                L2Norm()]
        elif role == "generative":
            encoder_layers += [NormalDistDecoder(num_neurons, latentD) ]
        elif role == "modifier":
            encoder_layers += [nn.Linear(num_neurons, latentD)]
        else:
            raise NotImplementedError

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, pose):
        return self.encoder(pose)

class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(dim=-1, keepdim=True)


class Object(object):
    pass

class TransformerTextEncoder(nn.Module):

    def __init__(self, text_encoder_name, num_neurons=512, latentD=512, topping=None, role=None,
                nlayers=4, nhead=4, dim_feedforward=1024, activation="gelu", dropout=0.1): # include parameters for the transformer topping
        super().__init__()

        self.role = role

        # load pretrained model weights & config
        self.using_pretrained_transformer = True # init
        self.pretrained_text_encoder = AutoModel.from_pretrained("cache/distilbert-base-uncased")

        if self.using_pretrained_transformer:
            print(f"Loaded text encoder pretrained weights ({text_encoder_name}).")
            # freeze pretrained model weights
            for param in self.pretrained_text_encoder.parameters():
                param.requires_grad = False
            # get embedding size
            txt_enc_last_dim = self.pretrained_text_encoder.config.hidden_size

        # learnable projection
        embed_dim = {"retrieval": latentD, "generative": num_neurons, "modifier": latentD}[role] # type: ignore
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(txt_enc_last_dim, embed_dim)) # type: ignore

        # define learnable transformer
        self.positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=0.1)
        transformer_encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout,
                                                    activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layers, nlayers)

        # define a way to represent the whole sequence from its token embeddings
        self.output_layer = None
        # - use average pooling
        if topping == "avgp":
            self.forward_topping = self.topping_avgp
            if role == "generative":
                self.output_layer = NormalDistDecoder(embed_dim, latentD)
        # - use learnable tokens
        elif topping == "augtokens":
            self.forward_topping = self.topping_augtokens
            nb_augm_tokens = {"retrieval":1, "generative":2}[role] # type: ignore # one retrieval token, or 2 distribution tokens for generation: mu & logvar
            self.augm_tokens = nn.ParameterList([nn.Parameter(torch.randn(embed_dim)) for i in range(nb_augm_tokens)])
            self.augm_token_final_FC_layers = nn.ModuleList([nn.Linear(embed_dim, latentD) for i in range(nb_augm_tokens)])
            if role == "generative":
                self.output_layer = torch.distributions.normal.Normal
        else:
            raise NotImplementedError

        if role == "retrieval":
            self.output_layer = L2Norm()
        elif role == "modifier":
            self.output_layer = nn.Sequential()

    def get_attention_mask(self, captions, caption_lengths):
        batch_size = len(captions)
        attention_mask = torch.zeros(batch_size, max(caption_lengths), device=captions.device).long()
        for i in range(batch_size):
            attention_mask[i, :caption_lengths[i]] = 1
        return attention_mask

    def average_pooling(self, token_embeddings, attention_mask):
        # take attention mask into account for correct mean pooling of all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        x = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return x

    def topping_avgp(self, token_embeddings, attention_mask):
        x = token_embeddings.permute(1, 0, 2) # (nbtokens, batch_size, latentID)
        # add positional encoding, pass through transformer
        x = self.positional_encoding(x)
        # pass through the learnable transformer
        r = self.transformer_encoder(x, src_key_padding_mask=~attention_mask.to(dtype=bool))
        # average pooling
        r = r.permute(1, 0, 2) # (batch_size, nbtokens, embed_dim)
        output = self.average_pooling(r, attention_mask)
        return self.output_layer(output) # type: ignore

    def topping_augtokens(self, token_embeddings, attention_mask):
        # add the augmentation tokens, for each element of the batch
        batch_size = token_embeddings.shape[0]
        x_augm = token_embeddings.permute(1, 0, 2) # (nbtokens, batch_size, latentID)
        for i_tok in range(len(self.augm_tokens) - 1, -1, -1): # consider tokens in reverse order, so that they are stored at the leftmost of the sequence, in the same order as in augm_tokens
            token_tile = torch.tile(self.augm_tokens[i_tok], (batch_size,)).reshape(batch_size, -1)
            x_augm = torch.cat((token_tile[None], x_augm), 0)
        # adapt the attention mask to account for the augmentation tokens
        dist_token_mask = torch.ones((batch_size, len(self.augm_tokens)), dtype=bool, device=x_augm.device) # type: ignore
        mask_augm = torch.cat((dist_token_mask, attention_mask.to(dtype=bool)), 1)
        # add positional encoding
        x_augm = self.positional_encoding(x_augm)
        # pass through the learnable transformer
        r = self.transformer_encoder(x_augm, src_key_padding_mask=~mask_augm)
        # extract final augmentation tokens
        output = [ self.augm_token_final_FC_layers[i](r[i]) for i in range(len(self.augm_tokens)) ]
        # return output
        if self.role == "retrieval":
            return self.output_layer(output[0]) # type: ignore # L2 norm
        elif self.role == "generative":
            return self.output_layer(output[0], output[1].exp().pow(0.5)) # type: ignore

    def forward(self, captions, caption_lengths):
        attention_mask = self.get_attention_mask(captions, caption_lengths)
        # embed tokens
        if self.using_pretrained_transformer:
            token_embeddings = self.pretrained_text_encoder(input_ids=captions, attention_mask=attention_mask).last_hidden_state
        else:
            token_embeddings = self.embed_sequential(captions)
        token_embeddings = self.projection(token_embeddings) # (batch_size, nbtokens, latentID)
        # apply transformer & topping
        return self.forward_topping(token_embeddings, attention_mask)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformTokenizer:
    def __init__(self):
          
        self.cased_tokenizer = False
        self.tokenizer = AutoTokenizer.from_pretrained("cache/distilbert-base-uncased")

        # define required token ids
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = 101 
        self.eos_token_id = 102 
        self.unk_token_id = self.tokenizer.unk_token_id
        self.max_tokens = self.tokenizer.model_max_length

    def __call__(self, text):
        x = self.tokenizer(text, truncation=True, return_tensors="pt")["input_ids"][0]
        return x
    
    def __len__(self):
        return len(self.tokenizer)
    
class ConCatModule(nn.Module):

    def __init__(self):
        super(ConCatModule, self).__init__()

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return x


# Pose AB retrieval


class PairText(nn.Module):
    def __init__(self, num_neurons=512, num_neurons_mini=32, latentD=512,
              text_encoder_name='distilbertUncased', transformer_topping=None):
        super(PairText, self).__init__()

        self.latentD = latentD

        # Define pose encoder
        self.pose_encoder = PoseEncoder(num_neurons, num_neurons_mini, latentD=latentD, role="retrieval")

        # Define text encoder
        self.text_encoder_name = text_encoder_name
        self.text_encoder = TransformerTextEncoder(self.text_encoder_name, latentD=latentD, topping=transformer_topping, role="retrieval")

        # Define projecting layers
        self.pose_mlp = nn.Sequential(
            ConCatModule(),
            nn.Linear(2 * latentD, 2 * latentD),
            nn.LeakyReLU(),
            nn.Linear(2 * latentD, latentD),
            nn.LeakyReLU(),
            nn.Linear(latentD, latentD),
            nn.LeakyReLU(),
            L2Norm()
        )

        # Loss temperature
        self.loss_weight = torch.nn.Parameter( torch.FloatTensor((10,)) )
        self.loss_weight.requires_grad = True

    def forward(self, poses_A, captions, caption_lengths, poses_B):
        embed_AB = self.encode_pose_pair(poses_A, poses_B)
        text_embs = self.encode_text(captions, caption_lengths)
        return embed_AB, text_embs

    def encode_raw_text(self, raw_text):
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = Tokenizer(get_tokenizer_name(self.text_encoder_name))
        tokens = self.tokenizer(raw_text).to(device=self.loss_weight.device)
        length = torch.tensor([ len(tokens) ], dtype=tokens.dtype)
        text_embs = self.text_encoder(tokens.view(1, -1), length)
        return text_embs
    
    def encode_pose_pair(self, poses_A, poses_B):
        embed_poses_A = self.pose_encoder(poses_A)
        embed_poses_B = self.pose_encoder(poses_B)
        embed_AB = self.pose_mlp([embed_poses_A, embed_poses_B])
        return embed_AB
    
    def encode_text(self, captions, caption_lengths):
        return self.text_encoder(captions, caption_lengths)