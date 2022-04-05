#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import socket
import random
import itertools
import numpy as np
import multiprocessing
import configparser as cp
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

import torch

np.random.seed(0)

def get_model_params(lr, first_additional_triplet, second_additional_triplet,  reg_loss, additional_triplets_loss, dropout_encoder, dropout_decoder, additional_dropout, encoder_hidden_size, decoder_hidden_size, depth_transformer, momentum):
    # Model parameters
    params_model = dict()
    params_model['dim_out'] = 64
    params_model['lr'] = lr
    if encoder_hidden_size==0:
        encoder_hidden_size=None
    if decoder_hidden_size==0:
        decoder_hidden_size=None
    params_model['first_additional_triplet'] = first_additional_triplet
    params_model['second_additional_triplet'] = second_additional_triplet
    params_model['additional_triplets_loss']=additional_triplets_loss
    params_model['additional_dropout'] = additional_dropout
    params_model['reg_loss']=reg_loss
    params_model['depth_transformer']=depth_transformer
    params_model['dropout_encoder']=dropout_encoder
    params_model['dropout_decoder']=dropout_decoder
    params_model['encoder_hidden_size']=encoder_hidden_size
    params_model['decoder_hidden_size']=decoder_hidden_size
    params_model['momentum']=momentum
    return params_model
