# import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import h5py


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()


class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K' or opt.dataset == 'smallImageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.opt = opt

    # not tested
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature'][()]
        label = fid['label'][()]
        trainval_loc = fid['trainval_loc'][()]
        train_loc = fid['train_loc'][()]
        val_unseen_loc = fid['val_unseen_loc'][()]
        test_seen_loc = fid['test_seen_loc'][()]
        test_unseen_loc = fid['test_unseen_loc'][()]
        fid.close()
        # read attributes
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".hdf5", 'r')
        self.attribute = fid['attribute'][()]
        fid.close()

        if not opt.validation:
            self.train_feature = feature[trainval_loc]
            self.train_label = label[trainval_loc]
            self.test_unseen_feature = feature[test_unseen_loc]
            self.test_unseen_label = label[test_unseen_loc]
            self.test_seen_feature = feature[test_seen_loc]
            self.test_seen_label = label[test_seen_loc]
        else:
            self.train_feature = feature[train_loc]
            self.train_label = label[train_loc]
            self.test_unseen_feature = feature[val_unseen_loc]
            self.test_unseen_label = label[val_unseen_loc]

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.nclasses = self.seenclasses.size(0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            train_feature_all = scaler.fit_transform(np.array(matcontent['train_features']))
            train_label_all = np.array(matcontent['train_labels']).astype(int).squeeze() - 1
            test_feature = scaler.transform(np.array(matcontent['test_features']))
            test_label = np.array(matcontent['test_labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            train_feature_all = np.array(matcontent['train_features'])
            train_label_all = np.array(matcontent['train_labels']).astype(int).squeeze() - 1
            test_feature = np.array(matcontent['test_features'])
            test_label = np.array(matcontent['test_labels']).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_fewshot_splits.mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        if opt.validation:
            self.baseclasses = torch.from_numpy(np.array(matcontent['base_classes1']).astype(int).squeeze()).long() - 1
            self.novelclasses = torch.from_numpy(
                np.array(matcontent['novel_classes1']).astype(int).squeeze()).long() - 1
            test_idx_baseclasses = torch.from_numpy(
                np.array(matcontent['idx_baseclasses1']).astype(int).squeeze()).long() - 1
            test_idx_novelclasses = torch.from_numpy(
                np.array(matcontent['idx_novelclasses1']).astype(int).squeeze()).long() - 1
        else:
            baseclasses2 = torch.from_numpy(np.array(matcontent['base_classes2']).astype(int).squeeze()).long() - 1
            self.baseclasses2 = baseclasses2
            baseclasses1 = torch.from_numpy(np.array(matcontent['base_classes1']).astype(int).squeeze()).long() - 1
            self.baseclasses = torch.cat((baseclasses1, baseclasses2), 0)
            self.test_baseclasses = baseclasses2
            self.novelclasses = torch.from_numpy(
                np.array(matcontent['novel_classes2']).astype(int).squeeze()).long() - 1
            test_idx_baseclasses = torch.from_numpy(
                np.array(matcontent['test_base2_loc']).astype(int).squeeze()).long() - 1
            test_idx_novelclasses = torch.from_numpy(
                np.array(matcontent['test_novel2_loc']).astype(int).squeeze()).long() - 1

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/sample_idx_split" + opt.splitid + ".mat")
        sample_idx_split = np.array(matcontent['sample_idx_split']).astype(int).squeeze()
        self.novel_idx = np.sort(sample_idx_split[self.novelclasses, :opt.kshot].reshape(-1))
        base_idx = np.array([]).astype(int)
        for c in self.baseclasses:
            cid = np.where(np.in1d(train_label_all, c))[0]
            num_select = round(len(cid) * opt.data_portion)
            base_idx = np.concatenate((base_idx, cid[:num_select]))
        self.base_idx = base_idx
        # self.base_idx = np.where(np.in1d(train_label_all, self.baseclasses))[0]

        train_base_feature = train_feature_all[self.base_idx, :]
        train_novel_feature = train_feature_all[self.novel_idx, :]
        train_base_label = train_label_all[self.base_idx]
        train_novel_label = train_label_all[self.novel_idx]
        test_base_feature = test_feature[test_idx_baseclasses, :]
        test_base_label = test_label[test_idx_baseclasses]
        test_novel_feature = test_feature[test_idx_novelclasses, :]
        test_novel_label = test_label[test_idx_novelclasses]

        self.test_base_feature = torch.from_numpy(test_base_feature).float()
        self.test_base_label = torch.from_numpy(test_base_label).long()
        self.test_novel_feature = torch.from_numpy(test_novel_feature).float()
        self.test_novel_label = torch.from_numpy(test_novel_label).long()

        self.test_feature = torch.from_numpy(test_feature).float()
        self.test_label = torch.from_numpy(test_label).long()

        self.train_feature = torch.from_numpy(np.concatenate((train_base_feature, train_novel_feature), axis=0)).float()
        self.train_label = torch.from_numpy(np.concatenate((train_base_label, train_novel_label), axis=0)).long()
        self.train_base_feature = torch.from_numpy(train_base_feature).float()
        self.train_base_label = torch.from_numpy(train_base_label).long()
        self.train_novel_feature = torch.from_numpy(train_novel_feature).float()
        self.train_novel_label = torch.from_numpy(train_novel_label).long()
        self.ntrain = self.train_feature.size()[0]
        self.ntest = self.test_novel_feature.size()[0]
        self.total_baseclass = self.baseclasses.size(0)
        self.total_novelclass = self.novelclasses.size(0)
        self.train_class = torch.cat((self.baseclasses, self.novelclasses), 0)
        self.ntrain_class = self.train_class.size(0)

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        # numpy array index starts from 0, matlab starts from 1
        train_base_loc = matcontent['train_base_loc'].squeeze() - 1
        test_base_loc = matcontent['test_base_loc'].squeeze() - 1
        test_novel_loc = matcontent['test_novel_loc'].squeeze() - 1
        novelclasses = np.unique(label[test_novel_loc])

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/sample_idx_split" + opt.splitid + ".mat")
        sample_idx_split = np.array(matcontent['sample_idx_split']).astype(int).squeeze() - 1
        train_novel_loc = np.sort(sample_idx_split[novelclasses, :opt.kshot].reshape(-1))
        train_loc = np.concatenate((train_base_loc, train_novel_loc)).astype(int)
        num_train_base_loc = train_base_loc.shape[0]

        # if opt.kshot > 0:
        #     novelclasses = np.unique(label[test_unseen_loc])
        #     kshot_novel_loc = np.array([]);
        #     test_novel_loc = np.array([])
        #     for c in novelclasses:
        #         idx_c = np.where(label == c)
        #         idx_c = np.array(idx_c[0])
        #         perm = np.random.permutation(len(idx_c))
        #         kshot_novel_loc = np.concatenate((kshot_novel_loc, idx_c[perm[0:opt.kshot]])).astype(int);
        #         test_novel_loc = np.concatenate((test_novel_loc, idx_c[perm[opt.kshot:]])).astype(int);
        #
        #     trainval_loc = np.concatenate((trainval_loc, kshot_unseen_loc)).astype(int)

        # if opt.image_att10:
        #     fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/image_text10.h5", 'r')
        #     image_att = fid['text_embedding'][()]
        #     image_att = torch.from_numpy(image_att)
        #     image_att = image_att.view(-1,10,opt.attSize)
        #     #idx = torch.LongTensor(len(trainval_loc)*10)
        #     #count = 0
        #     #for loc in trainval_loc:
        #     #    for i in range(10):
        #     #        idx[count] = loc.item()*10 + i
        #     #        count = count + 1
        #     loc = torch.LongTensor(len(trainval_loc))
        #     for i in range(len(trainval_loc)):
        #         loc[i] = trainval_loc[i].item()
        #     self.train_att = image_att[loc]
        #     loc2 = torch.LongTensor(len(test_unseen_loc))
        #     for i in range(len(test_unseen_loc)):
        #         loc2[i] = test_unseen_loc[i].item()
        #     self.test_att = image_att[loc2]
        #     fid.close()

        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[train_loc])
                _train_novel_feature = scaler.transform(feature[train_novel_loc])
                _train_base_feature = scaler.transform(feature[train_base_loc])
                _test_base_feature = scaler.transform(feature[test_base_loc])
                _test_novel_feature = scaler.transform(feature[test_novel_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(label[train_loc]).long()
                self.test_novel_feature = torch.from_numpy(_test_novel_feature).float()
                self.test_novel_feature.mul_(1 / mx)
                self.test_novel_label = torch.from_numpy(label[test_novel_loc]).long()
                self.test_base_feature = torch.from_numpy(_test_base_feature).float()
                self.test_base_feature.mul_(1 / mx)
                self.test_base_label = torch.from_numpy(label[test_base_loc]).long()
                self.train_novel_feature = torch.from_numpy(_train_novel_feature).float()
                self.train_novel_feature.mul_(1 / mx)
                self.train_novel_label = torch.from_numpy(label[train_novel_loc]).long()
                self.train_base_feature = torch.from_numpy(_train_base_feature).float()
                self.train_base_feature.mul_(1 / mx)
                self.train_base_label = torch.from_numpy(label[train_base_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_novel_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_novel_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_base_feature = torch.from_numpy(feature[test_seen_loc]).float()
                self.test_base_label = torch.from_numpy(label[test_seen_loc]).long()
                self.train_novel_feature = torch.from_numpy(feature[kshot_unseen_loc]).float()
                self.train_novel_label = torch.from_numpy(label[kshot_unseen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_novel_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_novel_label = torch.from_numpy(label[val_unseen_loc]).long()

        self.train_feature[num_train_base_loc:] = opt.novel_weight * self.train_feature[num_train_base_loc:]
        self.test_feature = torch.cat((self.test_base_feature, self.test_novel_feature), 0)
        self.test_label = torch.cat((self.test_base_label, self.test_novel_label), 0)
        self.baseclasses = torch.from_numpy(np.unique(self.train_base_label.numpy()))
        self.test_baseclasses = self.baseclasses
        self.novelclasses = torch.from_numpy(np.unique(self.train_novel_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntest = self.test_novel_feature.size()[0]
        self.ntrain_class = self.baseclasses.size(0) + self.novelclasses.size(0)
        self.train_class = torch.cat((self.baseclasses, self.novelclasses), 0)
        self.ntest_class = self.novelclasses.size(0)

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]

    def next_batch_unpair_test(self, batch_size):
        idx1 = torch.randperm(self.ntest)[0:batch_size]
        idx2 = torch.randperm(self.ntest)[0:batch_size]
        batch_feature = self.test_novel_feature[idx1]
        batch_label = self.test_novel_label[idx2]
        if self.opt.image_att:
            batch_att = self.test_att[idx2]
        else:
            batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        if self.opt.image_att:
            batch_att = self.train_att[idx]
        elif self.opt.image_att10:
            batch_att = torch.FloatTensor(batch_size, self.opt.attSize)
            for i in range(batch_size):
                idx2 = torch.randperm(10)[0]
                batch_att[i] = self.train_att[idx[i]][idx2]
                batch_att[i] = batch_att[i] / batch_att[i].norm()
        else:
            batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    # select batch samples by randomly drawing batch_size classes
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]

        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze(1)
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]]
        return batch_feature, batch_label, batch_att