import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import src.util_fewshot
import copy
from sklearn.preprocessing import MinMaxScaler 
import sys
import torch.nn.functional as F
class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, mapping_dict, data_loader, target_classes, _cuda, _lr=0.001, _beta1=0.5, _nepoch=30, _batch_size=1024, generalized=False, classes_gzsl=None):
        self.train_X =  _train_X 
        self.train_Y = _train_Y
        self.target_classes=target_classes
        self.mapping_dict=mapping_dict
        self.dataloader=data_loader
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model =  LINEAR_LOGSOFTMAX(self.input_dim, target_classes.shape[0])
        self.model.apply(src.util_fewshot.weights_init)
        self.criterion = nn.NLLLoss()
        if generalized==True:
            self.unseen_classes_gzsl, self.seen_classes_gzsl=classes_gzsl
        
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        if generalized:
            self.acc_all, self.acc_base, self.acc_novel = self.fit_gfsl()
        else:
            self.acc = self.fit_fsl() 

    
    def fit_fsl(self):
        best_acc = 0
        perm = torch.randperm(self.ntrain)
        self.train_X = self.train_X[perm]
        self.train_Y = self.train_Y[perm]

        (test_audio, test_video, test_Y_original) = self.dataloader
        test_X=F.normalize(torch.cat((test_audio, test_video), axis=1))
        test_Y=copy.deepcopy(test_Y_original)
        for i in range(test_Y_original.shape[0]):
            current_class=test_Y_original[i].item()
            new_class=self.mapping_dict[current_class]
            test_Y[i]=new_class

        for i in range(self.train_Y.size(0)):
            current_class=self.train_Y[i].item()
            new_class=self.mapping_dict[current_class]
            self.train_Y[i]=new_class

        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = F.normalize(self.train_X[i:i+self.batch_size]), self.train_Y[i:i+self.batch_size]

                inputv = batch_input.cuda()
                labelv = batch_label.cuda()
                #for j in range(labelv.size(0)):
                #    current_class=labelv[j].item()
                #    new_class=self.mapping_dict[current_class]
                #    labelv[j]=new_class
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()


            acc = self.val(test_X, test_Y, self.target_classes)
            if best_acc < acc:
                best_acc = acc
                
        return best_acc

    def fit_gfsl(self):
        best_acc_all = 0

        perm = torch.randperm(self.ntrain)
        self.train_X = self.train_X[perm]
        self.train_Y = self.train_Y[perm]

        (test_audio, test_video, test_Y_original) = self.dataloader
        test_X = F.normalize(torch.cat((test_audio, test_video), axis=1))
        test_Y = copy.deepcopy(test_Y_original)
        for i in range(test_Y_original.shape[0]):
            current_class = test_Y_original[i].item()
            new_class = self.mapping_dict[current_class]
            test_Y[i] = new_class
        for i in range(self.train_Y.size(0)):
            current_class=self.train_Y[i].item()
            new_class=self.mapping_dict[current_class]
            self.train_Y[i]=new_class

        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = F.normalize(self.train_X[i:i + self.batch_size]), self.train_Y[i:i + self.batch_size]
                       
                inputv = batch_input.cuda()
                labelv = batch_label.cuda()
                #for j in range(labelv.size(0)):
                #    current_class = labelv[j].item()
                #    new_class = self.mapping_dict[current_class]
                #    labelv[j] = new_class
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            
            acc_all, acc_novel, acc_base = self.val_gfsl(test_X, test_Y, self.seen_classes_gzsl, self.unseen_classes_gzsl)
            if best_acc_all <= acc_all:
                best_acc_all = acc_all
                best_acc_base = acc_base
                best_acc_novel = acc_novel
        return best_acc_all, best_acc_base, best_acc_novel
                     
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]
            
        batch_feature = torch.FloatTensor(batch_size, self.train_X.size(1))       
        batch_label = torch.LongTensor(batch_size)
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_Y.eq(iclass).nonzero().squeeze(1)
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_X[idx_file]
            batch_label[i] = self.train_Y[idx_file]
        return batch_feature, batch_label

    def val_gfsl_top1_top5(self, test_X, test_label, novelclasses, baseclasses): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        top1 = None
        top5 = None
        all_labels = None
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    output = self.model(test_X[start:end].cuda()) 
                else:
                    output = self.model(test_X[start:end]) 
            top1_this, top5_this = self.perelement_accuracy(output.data, test_label[start:end])
            top1 = top1_this if top1 is None else np.concatenate((top1, top1_this))
            top5 = top5_this if top5 is None else np.concatenate((top5, top5_this))
            all_labels = test_label[start:end].numpy() if all_labels is None else np.concatenate((all_labels, test_label[start:end].numpy()))
            start = end

        is_novel = np.in1d(all_labels, novelclasses)
        is_base = np.in1d(all_labels, baseclasses)
        is_either = is_novel | is_base
        top1_novel = np.mean(top1[is_novel])
        top1_base = np.mean(top1[is_base])
        top1_all = np.mean(top1[is_either])
        top5_novel = np.mean(top5[is_novel])
        top5_base = np.mean(top5[is_base])
        top5_all = np.mean(top5[is_either])
        return np.array([top1_novel, top5_novel, top1_base, top5_base, top1_all, top5_all])

    def val_gfsl(self, test_X, test_label, baseclasses, novelclasses): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    output = self.model(test_X[start:end].cuda()) 
                else:
                    output = self.model(test_X[start:end])
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc_base = self.compute_per_class_acc_gfsl(test_label, predicted_label, baseclasses )
        acc_novel =self.compute_per_class_acc_gfsl(test_label, predicted_label, novelclasses)
        acc_all=(2 * acc_novel * acc_base) / ((acc_novel + acc_base) + np.finfo(float).eps)
        return acc_all, acc_novel, acc_base

    def compute_per_class_acc_gfsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]).item() / torch.sum(idx).item()
        acc_per_class /= target_classes.shape[0]
        return acc_per_class

    def perelement_accuracy(self, scores, labels):
        topk_scores, topk_labels = scores.topk(5, 1, True, True)
        label_ind = labels.cpu().numpy()
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = topk_ind[:,0] == label_ind
        top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
        return top1_correct.astype(float), top5_correct.astype(float)

    # test_label is integer 
    def val_fsl_top1_top5(self, test_X, test_label): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        top1 = None
        top5 = None
        all_labels = None
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    output = self.model(test_X[start:end].cuda()) 
                else:
                    output = self.model(test_X[start:end]) 
            top1_this, top5_this = self.perelement_accuracy(output.data, test_label[start:end])
            top1 = top1_this if top1 is None else np.concatenate((top1, top1_this))
            top5 = top5_this if top5 is None else np.concatenate((top5, top5_this))
            all_labels = test_label[start:end].numpy() if all_labels is None else np.concatenate((all_labels, test_label[start:end].numpy()))
            start = end

        acc_top1 = np.mean(top1)
        acc_top5 = np.mean(top5)
        return acc_top1, acc_top5

    # test_label is integer 
    def val(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    output = self.model(test_X[start:end].cuda()) 
                else:
                    output = self.model(test_X[start:end]) 
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        #acc = torch.sum(test_label == predicted_label).item() / test_label.size(0)
        acc = self.compute_per_class_acc(test_label, predicted_label, target_classes.shape[0])
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).item() / torch.sum(idx).item()
        return acc_per_class.mean().item() 

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o  
