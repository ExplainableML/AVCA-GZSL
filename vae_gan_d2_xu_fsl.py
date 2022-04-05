from __future__ import print_function
import torch.nn.functional as F
import copy
import sys
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import math
from torch.utils import data
import src.classifier_fsl
import src.model_gen
import numpy as np
from src.dataset import ActivityNetDataset, AudioSetZSLDataset, ContrastiveDataset, VGGSoundDataset, UCFDataset
from src.metrics import DetailedLosses, MeanClassAccuracy, PercentOverlappingClasses, TargetDifficulty
from src.model_improvements import AVCA
from src.sampler import SamplerFactory
from src.train import train
from src.utils import fix_seeds, setup_experiment
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.args_gen import args_main
from src.utils_improvements import get_model_params

parser = argparse.ArgumentParser()
opt = args_main()

opt.nz = opt.latent_size
print(opt)


def fix_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


fix_seeds()

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# initialize generator and discriminator
netG = src.model_gen.Decoder(opt.decoder_layer_sizes, opt.latent_size, opt.attSize)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = src.model_gen.MLP_CRITIC(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

netD2 = src.model_gen.MLP_CRITIC_V(opt)
if opt.netD2 != '':
    netD2.load_state_dict(torch.load(opt.netD2))
print(netD2)

Encoder = src.model_gen.Encoder(opt.encoder_layer_sizes, opt.latent_size, opt.attSize)
if opt.Encoder != '':
    Encoder.load_state_dict(torch.load(opt.Encoder))
print(Encoder)

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
input_res_unpair = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att_unpair = torch.FloatTensor(opt.batch_size, opt.attSize)

if opt.cuda:
    netD.cuda()
    netD2.cuda()
    netG.cuda()
    Encoder.cuda()
    noise = noise.cuda()
    input_res = input_res.cuda()
    input_att = input_att.cuda()
    input_res_unpair = input_res_unpair.cuda()
    input_att_unpair = input_att_unpair.cuda()
    one = one.cuda()
    mone = mone.cuda()


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD


def sample():
    batch_feature, batch_label, batch_att = data.next_batch_uniform_class(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    batch_feature, batch_label, batch_att = data.next_batch_unpair_test(opt.batch_size)
    input_res_unpair.copy_(batch_feature)
    input_att_unpair.copy_(batch_att)


def generate_syn_feature(vae, classes, attribute, num, mapping):
    nclass = classes.shape[0]
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            mapped_class = mapping[iclass]
            iclass_att = attribute[mapped_class]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            output = netG(syn_noise, syn_att)
            syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label


# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerE = optim.Adam(Encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD2 = optim.Adam(netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates, input_att)

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


def calc_gradient_penalty2(netD, real_data, fake_data):
    # print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


# train a classifier on seen classes, obtain \theta of Equation (4)

if opt.dataset_name == "AudioSetZSL":
    train_dataset = AudioSetZSLDataset(
        args=opt,
        dataset_split="train",
        zero_shot_mode="seen",
    )

    val_dataset = AudioSetZSLDataset(
        args=opt,
        dataset_split="val",
        zero_shot_mode="seen",
    )

    train_val_dataset = AudioSetZSLDataset(
        args=opt,
        dataset_split="train_val",
        zero_shot_mode="seen",
    )

    val_all_dataset = AudioSetZSLDataset(
        args=opt,
        dataset_split="val",
        zero_shot_mode="all",
    )

elif opt.dataset_name == "VGGSound":
    train_dataset = VGGSoundDataset(
        args=opt,
        dataset_split="train",
        zero_shot_mode="train",
    )
    val_dataset = VGGSoundDataset(
        args=opt,
        dataset_split="val",
        zero_shot_mode=None,
    )

    train_val_dataset = VGGSoundDataset(
        args=opt,
        dataset_split="train_val",
        zero_shot_mode=None,
    )

    val_all_dataset = VGGSoundDataset(
        args=opt,
        dataset_split="val",
        zero_shot_mode=None,
    )
elif opt.dataset_name == "UCF":
    train_dataset = UCFDataset(
        args=opt,
        dataset_split="train",
        zero_shot_mode="train",
    )
    val_dataset = UCFDataset(
        args=opt,
        dataset_split="val",
        zero_shot_mode=None,
    )

    train_val_dataset = UCFDataset(
        args=opt,
        dataset_split="train_val",
        zero_shot_mode=None,
    )

    val_all_dataset = UCFDataset(
        args=opt,
        dataset_split="val",
        zero_shot_mode=None,
    )
elif opt.dataset_name == "ActivityNet":
    train_dataset = ActivityNetDataset(
        args=opt,
        dataset_split="train",
        zero_shot_mode="train",
    )
    val_dataset = ActivityNetDataset(
        args=opt,
        dataset_split="val",
        zero_shot_mode=None,
    )

    train_val_dataset = ActivityNetDataset(
        args=opt,
        dataset_split="train_val",
        zero_shot_mode=None,
    )

    val_all_dataset = ActivityNetDataset(
        args=opt,
        dataset_split="val",
        zero_shot_mode=None,
    )
else:
    raise NotImplementedError()

contrastive_train_dataset = ContrastiveDataset(train_dataset)
contrastive_val_dataset = ContrastiveDataset(val_dataset)
contrastive_train_val_dataset = ContrastiveDataset(train_val_dataset)
contrastive_val_all_dataset = ContrastiveDataset(val_all_dataset)
logger, log_dir, writer, train_stats, val_stats = setup_experiment(opt, "epoch", "loss", "hm")

train_sampler = SamplerFactory(logger).get(
    class_idxs=list(contrastive_train_dataset.target_to_indices.values()),
    batch_size=opt.bs,
    n_batches=opt.n_batches,
    alpha=1,
    kind='random'
)

val_sampler = SamplerFactory(logger).get(
    class_idxs=list(contrastive_val_dataset.target_to_indices.values()),
    batch_size=opt.bs,
    n_batches=opt.n_batches,
    alpha=1,
    kind='random'
)

train_val_sampler = SamplerFactory(logger).get(
    class_idxs=list(contrastive_train_val_dataset.target_to_indices.values()),
    batch_size=opt.bs,
    n_batches=opt.n_batches,
    alpha=1,
    kind='random'
)

val_all_sampler = SamplerFactory(logger).get(
    class_idxs=list(contrastive_val_all_dataset.target_to_indices.values()),
    batch_size=opt.bs,
    n_batches=opt.n_batches,
    alpha=1,
    kind='random'
)

train_loader = data.DataLoader(
    dataset=contrastive_train_dataset,
    batch_sampler=train_sampler,
    num_workers=2
)

val_loader = data.DataLoader(
    dataset=contrastive_val_dataset,
    batch_sampler=val_sampler,
    num_workers=2
)

train_val_loader = data.DataLoader(
    dataset=contrastive_train_val_dataset,
    batch_sampler=train_val_sampler,
    num_workers=2
)

val_all_loader = data.DataLoader(
    dataset=contrastive_val_all_dataset,
    batch_sampler=val_all_sampler,
    num_workers=2
)

if opt.dataset_name == "AudioSetZSL":
    val_all_dataset = AudioSetZSLDataset(
        args=opt,
        dataset_split="val",
        zero_shot_mode="all",
    )
    test_dataset = AudioSetZSLDataset(
        args=opt,
        dataset_split="test",
        zero_shot_mode="all",
    )
elif opt.dataset_name == "VGGSound":
    val_all_dataset = VGGSoundDataset(
        args=opt,
        dataset_split="val",
        # dataset_split="test",
        zero_shot_mode=None,
    )
    test_dataset = VGGSoundDataset(
        args=opt,
        dataset_split="test",
        zero_shot_mode=None,
    )
elif opt.dataset_name == "UCF":
    val_all_dataset = UCFDataset(
        args=opt,
        dataset_split="val",
        # dataset_split="test",
        zero_shot_mode=None,
    )
    test_dataset = UCFDataset(
        args=opt,
        dataset_split="test",
        zero_shot_mode=None,
    )
elif opt.dataset_name == "ActivityNet":
    val_all_dataset = ActivityNetDataset(
        args=opt,
        dataset_split="val",
        # dataset_split="test",
        zero_shot_mode=None,
    )
    test_dataset = ActivityNetDataset(
        args=opt,
        dataset_split="test",
        zero_shot_mode=None,
    )
else:
    raise NotImplementedError()

if opt.retrain_all:
    evaluation_dataset = test_dataset
    training_dataset = train_val_dataset
    training_dataloader = train_val_loader
else:
    evaluation_dataset = val_all_dataset
    training_dataset = train_dataset
    training_dataloader = train_loader

unseen_class_ids_list = list(evaluation_dataset.unseen_class_ids)
targets = evaluation_dataset.all_data['target']
index_array_val_unseen = torch.ones(targets.size(0), dtype=torch.bool)

for i in range(targets.size(0)):
    current_class = targets[i].item()
    if current_class not in unseen_class_ids_list:
        index_array_val_unseen[i] = 0

video_indexed = evaluation_dataset.all_data["video"][index_array_val_unseen]
audio_indexed = evaluation_dataset.all_data["audio"][index_array_val_unseen]
target_indexed = evaluation_dataset.all_data["target"][index_array_val_unseen]

best_acc_gzsl = 0
best_acc_zsl = 0
best_epoch = 0
for epoch in range(opt.nepoch):
    FP = 0
    mean_lossD = 0
    mean_lossG = 0

    for batch_idx, (data, target) in enumerate(training_dataloader):
        # import pdb; pdb.set_trace()
        p = data["positive"]
        x_p_a = p["audio"].cuda()
        x_p_v = p["video"].cuda()
        x_p_t = p["text"].cuda()
        x_p_num = target["positive"].cuda()

        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netD2.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG_v update

        # sample a mini-batch
        netD.zero_grad()
        netD2.zero_grad()
        # train with realG
        criticD_real = netD(F.normalize(torch.cat((x_p_a, x_p_v), axis=1).cuda()), x_p_t)
        criticD_real = criticD_real.mean()
        criticD_real.backward(mone)

        # non-conditional D on unpaired real data
        criticD_real_v_unpair_seen = netD2(F.normalize(torch.cat((x_p_a, x_p_v), axis=1).cuda()))
        criticD_real_v_unpair = opt.ud_weight * criticD_real_v_unpair_seen.mean()
        criticD_real_v_unpair.backward(mone)

        # train with fakeG
        noise.normal_(0, 1)
        fake = netG(noise, x_p_t)
        criticD_fake = netD(fake.detach(), x_p_t)
        criticD_fake = criticD_fake.mean()
        criticD_fake.backward(one)

        # non-conditional netD_v unpair fake data

        criticD_fake_v_unpair_seen = netD2(fake.detach())
        criticD_fake_v_unpair = opt.ud_weight * criticD_fake_v_unpair_seen.mean()

        criticD_fake_v_unpair.backward(one)

        # gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, F.normalize(torch.cat((x_p_a, x_p_v), axis=1)), fake, x_p_t)
        gradient_penalty.backward(retain_graph=True)

        # non-conditional D, gradient penalty
        gradient_penalty_v_unpair = opt.ud_weight * calc_gradient_penalty2(netD2, F.normalize(
            torch.cat((x_p_a, x_p_v), axis=1)), fake)
        gradient_penalty_v_unpair.backward()

        Wasserstein_D = criticD_real - criticD_fake
        D_cost = criticD_fake - criticD_real + gradient_penalty
        optimizerD.step()

        # non-conditional D, Wasserstein distance
        Wasserstein_D_v2 = criticD_real_v_unpair - criticD_fake_v_unpair
        D_cost_v2 = criticD_fake_v_unpair - criticD_real_v_unpair + gradient_penalty_v_unpair
        optimizerD2.step()
        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        if batch_idx % 5 == 0:
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = False  # avoid computation
            for p in netD2.parameters():  # reset requires_grad
                p.requires_grad = False  # they are set to False below in netG_v update

            netG.zero_grad()
            Encoder.zero_grad()
            # netG latent code vae loss
            mean, log_var = Encoder(F.normalize(torch.cat((x_p_a, x_p_v), axis=1)), x_p_t)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size]).cuda()
            z = eps * std + mean
            recon_x = netG(z, x_p_t)
            vae_loss = loss_fn(recon_x, F.normalize(torch.cat((x_p_a, x_p_v), axis=1)), mean, log_var)
            # netG latent code gan loss
            criticG_fake = netD(recon_x, x_p_t)
            criticG_fake = criticG_fake.mean()
            G_cost = -criticG_fake
            # net G fake data
            fake_v = netG(noise, x_p_t)
            criticG_fake2 = netD(fake_v, x_p_t)
            criticG_fake2 = criticG_fake2.mean()
            G_cost += -criticG_fake2
            # netG unpaired test data gan loss

            loss = opt.gan_weight * G_cost + opt.vae_weight * vae_loss
            loss.backward()
            optimizerG.step()
            optimizerE.step()
            break

    print('[%d/%d] Wasserstein_dist: %.4f, Wasserstein_dist2: %.4f, vae_loss:%.4f'
          % (epoch, opt.nepoch, Wasserstein_D.data.item(), Wasserstein_D_v2.data.item(), vae_loss.data.item()))

    # evaluate the model, set G to evaluation mode
    netG.eval()
    # Generalized few-shot learning

    w2v_emb_gzsl, map_dict_gzsl = evaluation_dataset.map_embeddings_target
    unseeen_class_ids_gzsl = evaluation_dataset.unseen_class_ids
    syn_feature_gzsl, syn_label_gzsl = generate_syn_feature(netG, unseeen_class_ids_gzsl, w2v_emb_gzsl, opt.syn_num,
                                                            map_dict_gzsl)

    train_features_gzsl = torch.cat((training_dataset.all_data["audio"], training_dataset.all_data["video"]), 1)
    train_labels_gzsl = training_dataset.all_data["target"]
    train_X = torch.cat((train_features_gzsl, syn_feature_gzsl), 0)
    train_Y = torch.cat((train_labels_gzsl, syn_label_gzsl), 0)
    final_map_dict_gzsl = {}
    counter = 0
    for i in range(train_Y.size(0)):
        current_class = train_Y[i].item()
        if current_class not in final_map_dict_gzsl:
            final_map_dict_gzsl[current_class] = counter
            counter += 1
    audio_indexed_gzsl = evaluation_dataset.all_data["audio"]
    video_indexed_gzsl = evaluation_dataset.all_data["video"]
    target_indexed_gzsl = evaluation_dataset.all_data["target"]

    all_classes = np.concatenate((evaluation_dataset.unseen_class_ids, training_dataset.seen_class_ids), axis=0)

    unseen_class_ids_gzsl_changed = evaluation_dataset.unseen_class_ids
    for i in range(unseen_class_ids_gzsl_changed.shape[0]):
        current_class = unseen_class_ids_gzsl_changed[i]
        new_class = final_map_dict_gzsl[current_class]
        unseen_class_ids_gzsl_changed[i] = new_class
    seen_class_ids_gzsl_changed = evaluation_dataset.seen_class_ids
    for i in range(seen_class_ids_gzsl_changed.shape[0]):
        current_class = seen_class_ids_gzsl_changed[i]
        new_class = final_map_dict_gzsl[current_class]
        seen_class_ids_gzsl_changed[i] = new_class

    cls = src.classifier_fsl.CLASSIFIER(train_X, train_Y, final_map_dict_gzsl,
                                        (audio_indexed_gzsl, video_indexed_gzsl, target_indexed_gzsl), all_classes,
                                        opt.cuda, opt.classifier_lr, 0.5, opt.nepoch_classifier, opt.syn_num, True,
                                        (unseen_class_ids_gzsl_changed, seen_class_ids_gzsl_changed))
    print('GZSL acc_all=', cls.acc_all, ', acc_base=', cls.acc_base, ', acc_novel=', cls.acc_novel)
    acc_gzsl = cls.acc_all

    # w2v_emb, map_dict=evaluation_dataset.map_embeddings_target
    unseen_class_ids = evaluation_dataset.unseen_class_ids
    # syn_feature, syn_label = generate_syn_feature(netG,  unseen_class_ids, w2v_emb, opt.syn_num, map_dict)
    final_map_dict = {}
    counter = 0
    for i in range(syn_label_gzsl.size(0)):
        current_class = syn_label_gzsl[i].item()
        if current_class not in final_map_dict:
            final_map_dict[current_class] = counter
            counter += 1

    cls = src.classifier_fsl.CLASSIFIER(syn_feature_gzsl, syn_label_gzsl, final_map_dict,
                                        (audio_indexed, video_indexed, target_indexed), unseen_class_ids, opt.cuda,
                                        opt.classifier_lr, 0.5, opt.nepoch_classifier, opt.syn_num, False)
    acc = cls.acc
    print('ZSL novel class accuracy= ', acc)
    # reset G to training mode
    if acc_gzsl > best_acc_gzsl:
        best_acc_gzsl = acc_gzsl
        best_acc_zsl = acc
        best_epoch = epoch
    netG.train()

print("Best epoch ", best_epoch, "with ZLS ", best_acc_zsl, "and GZSL", best_acc_gzsl)