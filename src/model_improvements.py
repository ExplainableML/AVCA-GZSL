#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F

# user defined
import src.utils_improvements

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, momentum,hidden_size=None):
        super(EmbeddingNet, self).__init__()
        modules = []
        if hidden_size:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size, momentum=momentum))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

# Inspired from https://github.com/AnjanDutta/sem-pcyc

class AVCA(nn.Module):
    def __init__(self, params_model, input_size_audio, input_size_video):
        super(AVCA, self).__init__()

        print('Initializing model variables...', end='')
        # Dimension of embedding
        self.dim_out = params_model['dim_out']
        # Number of classes
        self.hidden_size_encoder=params_model['encoder_hidden_size']
        self.hidden_size_decoder=params_model['decoder_hidden_size']
        self.r_enc=params_model['dropout_encoder']
        self.r_proj=params_model['dropout_decoder']
        self.depth_transformer=params_model['depth_transformer']
        self.additional_triplets_loss=params_model['additional_triplets_loss']
        self.reg_loss=params_model['reg_loss']
        self.r_dec=params_model['additional_dropout']
        self.momentum=params_model['momentum']

        self.first_additional_triplet=params_model['first_additional_triplet']
        self.second_additional_triplet=params_model['second_additional_triplet']

        print('Initializing trainable models...', end='')

        self.A_enc = EmbeddingNet(
            input_size=input_size_audio,
            hidden_size=self.hidden_size_encoder,
            output_size=300,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )
        self.V_enc = EmbeddingNet(
            input_size=input_size_video,
            hidden_size=self.hidden_size_encoder,
            output_size=300,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )
        self.cross_attention=Transformer(300, self.depth_transformer, 3, 100, 64, dropout=self.r_enc)

        self.W_proj= EmbeddingNet(
            input_size=300,
            output_size=self.dim_out,
            dropout=self.r_dec,
            momentum=self.momentum,
            use_bn=True
        )

        self.D = EmbeddingNet(
            input_size=self.dim_out,
            output_size=300,
            dropout=self.r_dec,
            momentum=self.momentum,
            use_bn=True
        )



        self.A_proj = EmbeddingNet(input_size=300, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)

        self.V_proj = EmbeddingNet(input_size=300, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)

        self.A_rec = EmbeddingNet(input_size=self.dim_out, output_size=300, dropout=self.r_dec, momentum=self.momentum, use_bn=True)

        self.V_rec = EmbeddingNet(input_size=self.dim_out, output_size=300, dropout=self.r_dec, momentum=self.momentum, use_bn=True)

        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, 300))

        # Optimizers
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']
        self.optimizer_gen = optim.Adam(list(self.A_proj.parameters()) + list(self.V_proj.parameters()) +
                                        list(self.A_rec.parameters()) + list(self.V_rec.parameters()) +
                                        list(self.V_enc.parameters()) + list(self.A_enc.parameters()) +
                                        list(self.cross_attention.parameters()) + list(self.D.parameters()) +
                                        list(self.W_proj.parameters()),
                                        lr=self.lr, weight_decay=1e-5)

        self.scheduler_gen =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, 'max', patience=3, verbose=True)

        print('Done')

        # Loss function
        print('Defining losses...', end='')
        self.criterion_reg = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        print('Done')

    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    def forward(self, audio, image, negative_audio, negative_image, word_embedding, negative_word_embedding):

        self.phi_a = self.A_enc(audio)
        self.phi_v = self.V_enc(image)

        self.phi_a_neg=self.A_enc(negative_audio)
        self.phi_v_neg=self.V_enc(negative_image)

        self.w=word_embedding
        self.w_neg=negative_word_embedding

        self.theta_w = self.W_proj(word_embedding)
        self.theta_w_neg=self.W_proj(negative_word_embedding)

        self.rho_w=self.D(self.theta_w)
        self.rho_w_neg=self.D(self.theta_w_neg)

        self.positive_input=torch.stack((self.phi_a + self.pos_emb1D[0, :], self.phi_v + self.pos_emb1D[1, :]), dim=1)
        self.negative_input=torch.stack((self.phi_a_neg + self.pos_emb1D[0, :], self.phi_v_neg + self.pos_emb1D[1, :]), dim=1)

        self.phi_attn= self.cross_attention(self.positive_input)

        self.phi_attn_neg = self.cross_attention(self.negative_input)

        self.audio_fe_attn = self.phi_a + self.phi_attn[:, 0, :]
        self.video_fe_attn= self.phi_v + self.phi_attn[:, 1, :]


        self.audio_fe_neg_attn = self.phi_a_neg + self.phi_attn_neg[:, 0, :]
        self.video_fe_neg_attn = self.phi_v_neg + self.phi_attn_neg[:, 1, :]

        self.theta_v = self.V_proj(self.video_fe_attn)
        self.theta_v_neg=self.V_proj(self.video_fe_neg_attn)
        self.theta_a = self.A_proj(self.audio_fe_attn)
        self.theta_a_neg=self.A_proj(self.audio_fe_neg_attn)

        self.phi_v_rec = self.V_rec(self.theta_v)
        self.phi_a_rec = self.A_rec(self.theta_a)
        self.se_em_hat1 = self.A_proj(self.phi_a_rec)
        self.se_em_hat2 = self.V_proj(self.phi_v_rec)


        self.rho_a=self.D(self.theta_a)
        self.rho_a_neg=self.D(self.theta_a_neg)
        self.rho_v=self.D(self.theta_v)
        self.rho_v_neg=self.D(self.theta_v_neg)


    def backward(self, optimize):

        if self.additional_triplets_loss==True:
            first_pair = self.first_additional_triplet*(self.triplet_loss(self.theta_a, self.theta_w, self.theta_a_neg) + \
                                                        self.triplet_loss(self.theta_v, self.theta_w, self.theta_v_neg))
            second_pair=self.second_additional_triplet*(self.triplet_loss(self.theta_w, self.theta_a, self.theta_w_neg) + \
                                                        self.triplet_loss(self.theta_w, self.theta_v, self.theta_w_neg))

            l_t=first_pair+second_pair

        if self.reg_loss==True:
            l_r = (self.criterion_reg(self.phi_v_rec, self.phi_v) + \
                            self.criterion_reg(self.phi_a_rec, self.phi_a) + \
                            self.criterion_reg(self.theta_v, self.theta_w) + \
                            self.criterion_reg(self.theta_a, self.theta_w))


        l_rec= self.criterion_reg(self.w, self.rho_v) + \
                  self.criterion_reg(self.w, self.rho_a) + \
                  self.criterion_reg(self.w, self.rho_w)

        l_ctv=self.triplet_loss(self.rho_w, self.rho_v, self.rho_v_neg)
        l_cta=self.triplet_loss(self.rho_w, self.rho_a, self.rho_a_neg)
        l_ct=l_cta+l_ctv
        l_cmd=l_rec+l_ct

        l_tv = self.triplet_loss(self.theta_w, self.theta_v, self.theta_v_neg)
        l_ta = self.triplet_loss(self.theta_w, self.theta_a, self.theta_a_neg)
        l_at = self.triplet_loss(self.theta_a, self.theta_w, self.theta_w_neg)
        l_vt = self.triplet_loss(self.theta_v, self.theta_w, self.theta_w_neg)

        l_w=l_ta+l_at+l_tv+l_vt

        loss_gen=l_cmd + l_w
        if self.additional_triplets_loss==True:
           loss_gen+=l_t
        if self.reg_loss==True:
            loss_gen+=l_r

        if optimize == True:
            self.optimizer_gen.zero_grad()
            loss_gen.backward()
            self.optimizer_gen.step()

        loss = {'aut_enc': 0,  'gen_cyc': 0,
                'gen_reg': 0, 'gen': loss_gen}

        loss_numeric = loss['gen_cyc'] + loss['gen']

        return loss_numeric, loss

    def optimize_params(self, audio, video, cls_numeric, cls_embedding,audio_negative, video_negative, negative_cls_embedding,optimize=False):

        self.forward(audio, video, audio_negative, video_negative, cls_embedding, negative_cls_embedding)

        loss_numeric, loss = self.backward(optimize)

        return loss_numeric, loss

    def get_embeddings(self, audio, video, embedding):

        phi_a = self.A_enc(audio)
        phi_v = self.V_enc(video)
        theta_w=self.W_proj(embedding)

        input_concatenated=torch.stack((phi_a+self.pos_emb1D[0,:], phi_v+self.pos_emb1D[1,:]), dim=1)

        phi_attn= self.cross_attention(input_concatenated)

        phi_a = phi_a + phi_attn[:,0,:]
        phi_v = phi_v + phi_attn[:,1,:]

        theta_v = self.V_proj(phi_v)
        theta_a = self.A_proj(phi_a)

        return theta_a, theta_v, theta_w


