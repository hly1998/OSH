from utils.tools import *
from network import ResNet_W
import torch
import torch.optim as optim
import time
import numpy as np
import random
import logging
import psutil
import logging
import kornia.augmentation as Kg
import torch.nn.functional as F
import pandas as pd

torch.multiprocessing.set_sharing_strategy('file_system')

def get_config():
    config = {
        "transformation_scale": 0.8,
        "center_dis": 1, 
        "ranking": 1,
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-3, "weight_decay": 10 ** -5}},
        "info": "ours",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "dataset": "cifar100",
        "epoch": 100,
        "test_map": 1,
        "eval_epoch": 0,
        "max_norm": 5.0,
        "device": torch.device("cuda:0"),
        "bit_list": [64], 
        "stop_iter": 7,
        "n_positive": 2,
    }
    config = config_dataset(config)
    return config

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

class ClassLoss(torch.nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.CELoss = torch.nn.CrossEntropyLoss()

    def forward(self, probs, labels):
        labels = labels.float()
        celoss = self.CELoss(probs, labels)
        return celoss

def top_k_accuracy(output, target, k=1):
    with torch.no_grad():
        _, predicted = torch.max(output.data, 1)
        _, target = torch.max(target, 1)
        total_correct = (predicted == target).sum().item()
        total = target.shape[0]
    return total_correct, total

def test_val(config, model, test_loader, device):
    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for img, label, ind in tqdm(test_loader):
            img = img.to(device)
            label = label.to(device)
            preds = model(img)
            temp_acc,  temp_batch = top_k_accuracy(preds, label, k=1)
            acc += temp_acc
            total += temp_batch
    return 100 * acc / total

def train_val(config, bit, backbone):
    logging.basicConfig(filename='./logs/' + config["dataset"] + '_' + config["info"] + '.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("create the log file...")
    device = config["device"]
    explore_train_loader, train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    class_net = ResNet_W(config['n_class']).to(device)

    optimizer = config["optimizer"]["type"]([class_net.parameters()], **(config["optimizer"]["optim_params"]))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=config["epoch"])
    Best_mAP = 0
    count = 0
    cross_entropy_loss = ClassLoss()

    AugS = Augmentation(config["resize_size"], 1.0)
    Crop = torch.nn.Sequential(Kg.CenterCrop(config["crop_size"]))
    Norm = torch.nn.Sequential(Kg.Normalize(mean=torch.as_tensor([0.485, 0.456, 0.406]), std=torch.as_tensor([0.229, 0.224, 0.225])))

    tmp_input = torch.ones([config['n_class'], 2048]).to(device)
    n_class = config['n_class']
    class_net.train()
    Best_acc = 0
    for epoch in range(config["epoch"]):
        train_acc = 0
        total = 0
        for image, label, ind in explore_train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            Is = Norm(Crop(AugS(image)))
            probs = class_net(Is)
            class_loss = cross_entropy_loss(probs, label)
            temp_acc, temp_batch= top_k_accuracy(probs, label, k=1)
            train_acc += temp_acc
            total += temp_batch
            class_loss.backward()
            torch.nn.utils.clip_grad_norm_(class_net.parameters(), config['max_norm'])
            optimizer.step()
        train_acc /= total
        print(f"train accuracy: {100 * train_acc}")
        if (epoch + 1) % config['test_map'] == 0:
            test_acc = test_val(config, class_net, test_loader, device)
            print(f"test accuracy: {test_acc}")
            if test_acc > Best_acc:
                Best_acc = test_acc
                count = 0
                print(f'a better model find, save...')
                torch.save(class_net, './checkpoints/classifier_model_' +  config["dataset"] + '_' + config["info"] + '_' + str(bit) + '.pt')
           

if __name__ == "__main__":
    config = get_config()
    seed = 43
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    for bit in config["bit_list"]:
        for net in config["net"]:
            train_val(config, bit, net)
