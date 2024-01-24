from utils.tools import *
from network_xnor_resnet import ResNet_XNOR_v9
from network import ResNet_W
import torch
import torch.optim as optim
import time
import numpy as np
import random
from model_backbone.layers_xnor import XNORLinear, XNORConv2d
import logging
import logging
from torch.autograd import Function
import kornia.augmentation as Kg
import torch.nn.functional as F
import pandas as pd
from model_backbone.network_ir import ResNet_IR
from torch.optim import lr_scheduler
torch.multiprocessing.set_sharing_strategy('file_system')

class hash(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def hash_layer(input):
    return hash.apply(input)

class Augmentation(torch.nn.Module):
    def __init__(self, org_size, Aw=1.0):
        super(Augmentation, self).__init__()
        self.gk = int(org_size*0.1)
        if self.gk%2==0:
            self.gk += 1
        self.Aug = torch.nn.Sequential(
        Kg.RandomResizedCrop(size=(org_size, org_size), p=1.0*Aw),
        Kg.RandomHorizontalFlip(p=0.5*Aw),
        Kg.ColorJitter(brightness=0.4, contrast=0.8, saturation=0.8, hue=0.2, p=0.8*Aw),
        Kg.RandomGrayscale(p=0.2*Aw),
        Kg.RandomGaussianBlur((self.gk, self.gk), (0.1, 2.0), p=0.5*Aw))

    def forward(self, x):
        return self.Aug(x)

class Label_func(torch.nn.Module):
    def __init__(self, N_bits, fc_dim=2048):
        super(Label_func, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(fc_dim, fc_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_dim, N_bits))

    def forward(self, X, W):
        w = self.mlp(W)
        w = torch.sigmoid(w)
        w = hash_layer(w - 0.5)
        x = X @ w
        return x, w

class HashDistill(torch.nn.Module):
    def __init__(self, batch_size, device, n_positive):
        super(HashDistill, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.n_positive = n_positive
    def forward(self, xS, xT):
        HKDloss = 0
        num_image = xS.shape[0]
        num_k = int(num_image / self.n_positive)
        for k in range(num_k):
            xs = xS[self.n_positive*k : self.n_positive*(k+1),:]
            xt = xT[self.n_positive*k : self.n_positive*(k+1),:]
            pos_mask = np.random.uniform(0.0, 1.0, size=2*self.n_positive)+1
            pos_mask = torch.from_numpy(pos_mask).float().to(self.device)
            xx = torch.cat((xs,xt), dim=0)
            xx = torch.mean(xx, dim=0).repeat(self.n_positive, 1)
            pos_mask = pos_mask.unsqueeze(-1)
            xs = pos_mask[:self.n_positive] * xs + (1 - pos_mask[:self.n_positive]) * xx
            xt = pos_mask[self.n_positive:] * xt + (1 - pos_mask[self.n_positive:]) * xx
            new_xx = torch.cat((xs,xt), dim=0)
            n = 2*self.n_positive
            norm_x = new_xx / new_xx.norm(dim=1, keepdim=True)
            cosine_similarity_matrix = torch.mm(norm_x, norm_x.t())
            triu_indices = torch.triu_indices(n, n, offset=1)
            cosine_similarities = cosine_similarity_matrix[triu_indices[0], triu_indices[1]]
            HKDloss += cosine_similarities.mean()
        HKDloss = HKDloss / self.batch_size
        return - HKDloss

class HashProxy(torch.nn.Module):
    def __init__(self, temp):
        super(HashProxy, self).__init__()
        self.temp = temp

    def forward(self, X, P, L):
        X = F.normalize(X, p = 2, dim = -1)
        P = F.normalize(P, p = 2, dim = -1)
        D = F.linear(X, P) / self.temp
        xent_loss = torch.mean(torch.sum(-L * F.log_softmax(D, -1), -1))
        return xent_loss

class ClassLoss(torch.nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.CELoss = torch.nn.CrossEntropyLoss()

    def forward(self, probs, labels):
        labels = labels.float()
        celoss = self.CELoss(probs, labels)
        return celoss

def train_val(config):
    logging.basicConfig(filename='./logs/' + config["dataset"] + '_' + config["info"] + '.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("create the log file...")
    device = config["device"]
    bit = config["bit"]
    explore_train_loader, train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    if config["net"]==ResNet_IR:
        net = config["net"](bit, config["device"]).to(device)
    else:
        net = config["net"](bit).to(device)
    label_net = Label_func(bit).to(device)
    class_net = torch.load('./checkpoints/classifier_model_' +  config["dataset"] + '_' + config["info"] + '_' + str(bit) + '.pt', map_location=device).to(device)
    for param in class_net.model_resnet.parameters():
        param.requires_grad = False
    optimizer = config["optimizer"]["type"](list(net.parameters()) + list(label_net.parameters()) + list(class_net.parameters()), **(config["optimizer"]["optim_params"]))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=config["epoch"])
    Best_mAP = 0
    count = 0
    n_positive = config["n_positive"]
    HP_criterion = HashProxy(config["temp"])
    HD_criterion = HashDistill(config["batch_size"], device, n_positive)
    cross_entropy_loss = ClassLoss()

    AugS = Augmentation(config["resize_size"], 1.0)
    AugT = Augmentation(config["resize_size"], config["transformation_scale"])

    Crop = torch.nn.Sequential(Kg.CenterCrop(config["crop_size"]))
    Norm = torch.nn.Sequential(Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225])))

    tmp_input = torch.ones([config['n_class'], 2048]).to(device)
    n_class = config['n_class']
    net.train()
    label_net.train()
    class_net.train()
    train_dis_list = []
    train_coh_list = []
    test_dis_list = []
    test_coh_list = []

    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        s_time = time.time()
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        label_net.train()
        class_net.train()
        train_loss = 0
        train_loss_l1 = 0
        train_loss_l2 = 0
        train_loss_l3 = 0
        if epoch < config["transition_epoch"]:
            for image, label, _ in explore_train_loader:
                image = image.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                It = Norm(Crop(AugT(image)))
                Xt = net(It).tanh()
                w = class_net(tmp_input, 'l')
                y, hash_centers = label_net(label.float(), w)
                l1 = HP_criterion(Xt, hash_centers, label)
                probs = class_net(It)
                l3 = cross_entropy_loss(probs, label) * config["lambda2"]
                loss = l1 + l3
                train_loss_l1 += l1.item()
                train_loss_l3 += l3.item()
                train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(label_net.parameters(), config['max_norm'])
                torch.nn.utils.clip_grad_norm_(class_net.parameters(), config['max_norm'])
                for m in net.modules():
                    if isinstance(m, XNORConv2d) or isinstance(m, XNORLinear): 
                        m.update_gradient()
                optimizer.step()
                scheduler.step()
            train_loss = train_loss / len(train_loader)
            train_loss_l1 = train_loss_l1 / len(train_loader)
            train_loss_l3 = train_loss_l3 / len(train_loader)
            print("\b\b\b\b\b\b\b space exploration stage complete-train loss:%.3f, train loss 1:%.3f, train loss 3:%.7f, cost time:%.3f" % (train_loss, train_loss_l1, train_loss_l3, time.time() - s_time))
            if (epoch + 1) % 25 == 0:
                Best_mAP_before = Best_mAP
                mAP, Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
                logging.info(f"{net.__class__.__name__} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
                if mAP > Best_mAP_before:
                    torch.save(net, './checkpoints/' + config["dataset"] + '_' + config["bnn_model"] + '_' + config["info"] + '_' + str(bit) + '.pt')
                    count = 0
                else:
                    if count == config['stop_iter']:
                        break
                    count += 1
        else:
            for image, label, _ in train_loader:
                c, h, w = image.size()[-3:]
                image = image.view(-1, c, h, w)
                l = label.size()[-1]
                label = label.view(-1, l)
                image = image.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                Is = Norm(Crop(AugS(image)))
                It = Norm(Crop(AugT(image)))
                Xs = net(Is).tanh()
                Xt = net(It).tanh()
                w = class_net(tmp_input, 'l')
                y, hash_centers = label_net(label.float(), w)
                l1 = HP_criterion(Xt, hash_centers, label)
                l2 = HD_criterion(Xs, Xt.detach()) * config["lambda1"]
                probs = class_net(Is)
                l3 = cross_entropy_loss(probs, label) * config["lambda2"]
                loss = l1 + l2 + l3
                train_loss_l1 += l1.item()
                train_loss_l2 += l2.item()
                train_loss_l3 += l3.item()
                train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(label_net.parameters(), config['max_norm'])
                torch.nn.utils.clip_grad_norm_(class_net.parameters(), config['max_norm'])
                for m in net.modules():
                    if isinstance(m, XNORConv2d) or isinstance(m, XNORLinear): 
                        m.update_gradient()
                optimizer.step()
                scheduler.step()
            train_loss = train_loss / len(train_loader)
            train_loss_l1 = train_loss_l1 / len(train_loader)
            train_loss_l2 = train_loss_l2 / len(train_loader)
            train_loss_l3 = train_loss_l3 / len(train_loader)
            print("\b\b\b\b\b\b\b code aggregation stage complete-train loss:%.3f, train loss 1:%.3f, train loss 2:%.6f, train loss 3:%.7f, cost time:%.3f" % (train_loss, train_loss_l1, train_loss_l2, train_loss_l3, time.time() - s_time))
            if (epoch + 1) % config["test_map"] == 0 and epoch >= config["eval_epoch"]:
                Best_mAP_before = Best_mAP
                mAP, Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
                logging.info(f"{net.__class__.__name__} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
                if mAP > Best_mAP_before:
                    torch.save(net, './checkpoints/' + config["dataset"] + '_' + config["bnn_model"] + '_' + config["info"] + '_' + str(bit) + '.pt')
                    count = 0
                else:
                    if count == config['stop_iter']:
                        break
                    count += 1