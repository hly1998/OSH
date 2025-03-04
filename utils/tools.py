import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import os
import json
import logging.config
import random

def config_dataset(config):
    if "cifar100" in config["dataset"]:
        config["topK"] = 500
        config["n_class"] = 100
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38
    elif config["dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20

    config["data_path"] = "/dataset/" + config["dataset"] + "/"
    if config["dataset"] == "imagenet":
        config["data_path"] = "/data/lyhe/image_data/imagenet/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "/data/lyhe/image_data/nuswide_21/"
    config["data"] = {
        "train_set": {"list_path": "/data/lyhe/image_data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "/data/lyhe/image_data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "/data/lyhe/image_data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config

class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

class ImageList_special(object):
    def __init__(self, data_path, image_list, transform, num_classes=100, n_positive=8):
        self.n_positive = n_positive
        self.imgs = [data_path + val.split()[0] for val in image_list]
        self.targets = [np.array([int(la) for la in val.split()[1:]]) for val in image_list]
        self.transform = transform
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(len(self.imgs)):
            cls_idx = np.argmax(self.targets[i])
            self.cls_positive[cls_idx].append(i)

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.targets[index]
        cls_idx = np.argmax(target)
        idxs = np.random.choice(self.cls_positive[cls_idx], self.n_positive, replace=False)
        imgs = []
        targets = []
        for idx in idxs:
            img = Image.open(self.imgs[idx]).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
            targets.append(self.targets[idx])
        imgs = np.stack(imgs)
        targets = np.stack(targets)
        return imgs, targets, index

    def __len__(self):
        return len(self.imgs)

def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])

def image_transform_v9(resize_size, crop_size, data_set):
    if data_set == "train_set":
        return transforms.Compose([transforms.Resize(resize_size),
                                   transforms.RandomCrop(crop_size),
                                   transforms.ToTensor()])
    else:
        step = [transforms.CenterCrop(crop_size)]
        return transforms.Compose([transforms.Resize(resize_size)]
                                + step +
                                [transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ])

def image_transform_ours(resize_size, crop_size, data_set):
    if data_set == "train_set":
        return transforms.Compose([transforms.Resize(resize_size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(crop_size),
                                   transforms.ToTensor()])
    else:
        step = [transforms.CenterCrop(crop_size)]
        return transforms.Compose([transforms.Resize(resize_size)]
                                + step +
                                [transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ])

class MyCIFAR100(dsets.CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(100, dtype=np.int8)[np.array(target)]
        return img, target, index

class MyCIFAR100_special(dsets.CIFAR100):
    def __init__(self, num_classes=100, n_positive=8, **kwargs):
        super().__init__(**kwargs)
        self.n_positive = n_positive
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(10000):
            cls_idx = self.targets[i]
            self.cls_positive[cls_idx].append(i)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        cls_idx = target
        idxs = np.random.choice(self.cls_positive[cls_idx], self.n_positive, replace=False)
        imgs = []
        targets = []
        for idx in idxs:
            img = Image.fromarray(self.data[idx])
            img = self.transform(img)
            imgs.append(img)
            targets.append(self.targets[idx])
        imgs = np.stack(imgs)
        targets = np.stack(targets)
        targets = np.eye(100, dtype=np.int8)[np.array(targets)]
        return imgs, targets, index


def cifar100_dataset(config):
    batch_size = config["batch_size"]

    train_size = 100
    test_size = 50

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cifar_dataset_root = '/data/lyhe/image_data/cifar100/'
    if config["info"] in ["ours"]:
        explore_train_dataset = MyCIFAR100(root=cifar_dataset_root,
                                train=True,
                                transform=image_transform_v9(config["resize_size"], config["crop_size"], "train_set"),
                                download=True)
        
        aggregation_train_dataset = MyCIFAR100_special(root=cifar_dataset_root,
                                train=True,
                                transform=image_transform_v9(config["resize_size"], config["crop_size"], "train_set"),
                                download=True, 
                                n_positive=config['n_positive'])
        
        test_dataset = MyCIFAR100(root=cifar_dataset_root,
                                train=False,
                                transform=image_transform_v9(config["resize_size"], config["crop_size"], "test"))

        database_dataset = MyCIFAR100(root=cifar_dataset_root,
                                    train=False,
                                    transform=image_transform_v9(config["resize_size"], config["crop_size"], "database"))
    
    else:
        explore_train_dataset = MyCIFAR100(root=cifar_dataset_root,
                                train=True,
                                transform=transform,
                                download=True)
        
        aggregation_train_dataset = MyCIFAR100_special(root=cifar_dataset_root,
                                train=True,
                                transform=transform,
                                download=True, 
                                n_positive=config['n_positive'])
        
        test_dataset = MyCIFAR100(root=cifar_dataset_root,
                                train=False,
                                transform=transform)

        database_dataset = MyCIFAR100(root=cifar_dataset_root,
                                    train=False,
                                    transform=transform)

    X = np.concatenate((explore_train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(explore_train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(100):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    database_index = np.concatenate((train_index, database_index))
    
    explore_train_dataset.data = X[train_index]
    explore_train_dataset.targets = L[train_index]
    aggregation_train_dataset.data = X[train_index]
    aggregation_train_dataset.targets = L[train_index]
    

    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", explore_train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    explore_train_loader = torch.utils.data.DataLoader(dataset=explore_train_dataset,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=6)
    
    aggregation_train_loader = torch.utils.data.DataLoader(dataset=aggregation_train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=6)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=6)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=64,
                                                  shuffle=False,
                                                  num_workers=6)

    return explore_train_loader, aggregation_train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]

def get_data(config):
    if "cifar" in config["dataset"]:
        return cifar100_dataset(config)
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        if config["info"] in ["ours"] and data_set == 'train_set':
            dsets["explore_set"] = ImageList(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform_v9(config["resize_size"], config["crop_size"], data_set))
            dsets[data_set] = ImageList_special(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform_v9(config["resize_size"], config["crop_size"], data_set), n_positive=config['n_positive'])
        elif config["info"] in ["ours"]:
            dsets[data_set] = ImageList(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform_v9(config["resize_size"], config["crop_size"], data_set))
        else:
            dsets[data_set] = ImageList(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform(config["resize_size"], config["crop_size"], data_set))

        print(data_set, len(dsets[data_set]))

        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle= (data_set == "train_set") , num_workers=12)
    if config["info"] in ["ours"]:
        dset_loaders["explore_set"] = util_data.DataLoader(dsets["explore_set"],
                                                      batch_size=64,
                                                      shuffle=True , num_workers=12)
        return dset_loaders["explore_set"], dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])
    else:
        return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
            len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        out = net(img.to(device))
        if isinstance(out, tuple):
            bs.append(out[0].data.cpu())
        else:
            bs.append((out).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_MDSH(dataloader, net, device, T, label_vector):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device), T, label_vector)[0]).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def CalcTopMap_for_v13(rB, qB, retrievalL, queryL, topk, weights):
    num_query = queryL.shape[0]
    topkmap = 0
    rB_weight = weights
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall

def validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset):
    device = config["device"]
    tst_binary, tst_label = compute_result(test_loader, net, device=device)
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

    if "pr_curve_path" not in  config:
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])
    else:
        mAP, cum_prec, cum_recall = CalcTopMapWithPR(tst_binary.numpy(), tst_label.numpy(),
                                                     trn_binary.numpy(), trn_label.numpy(),
                                                     config["topK"])
        index_range = num_dataset // 100
        index = [i * 100 - 1 for i in range(1, index_range + 1)]
        max_index = max(index)
        overflow = num_dataset - index_range * 100
        index = index + [max_index + i for i in range(1, overflow + 1)]
        c_prec = cum_prec[index]
        c_recall = cum_recall[index]

        pr_data = {
            "index": index,
            "P": c_prec.tolist(),
            "R": c_recall.tolist()
        }
        os.makedirs(os.path.dirname(config["pr_curve_path"]), exist_ok=True)
        with open(config["pr_curve_path"], 'w') as f:
            f.write(json.dumps(pr_data))
        print("pr curve save to ", config["pr_curve_path"])

    if mAP > Best_mAP:
        Best_mAP = mAP
        if "save_path" in config:
            save_path = os.path.join(config["save_path"], f'{config["dataset"]}_{bit}bits_{mAP}')
            os.makedirs(save_path, exist_ok=True)
            print("save in ", save_path)
            np.save(os.path.join(save_path, "tst_label.npy"), tst_label.numpy())
            np.save(os.path.join(save_path, "tst_binary.npy"), tst_binary.numpy())
            np.save(os.path.join(save_path, "trn_binary.npy"), trn_binary.numpy())
            np.save(os.path.join(save_path, "trn_label.npy"), trn_label.numpy())
            torch.save(net.state_dict(), os.path.join(save_path, "{}_model.pt".format(net.__class__.__name__)))
    print(f"epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
    print(config)
    return mAP, Best_mAP