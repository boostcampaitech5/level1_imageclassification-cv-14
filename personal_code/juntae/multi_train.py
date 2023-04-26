import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import wandb
import gc
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchsampler import ImbalancedDatasetSampler

from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

from collections import Counter

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    # cut_rat = np.sqrt(1. - lam)
    # cut_w = np.int(W * cut_rat)
    # cut_h = np.int(H * cut_rat)
    
    bbx1 = np.clip(0,0,W)
    bby1 = np.clip(H // 2,0,H)
    bbx2 = np.clip(W,0,W)
    bby2 = np.clip(H,0,H)

    # uniform
    # cx = np.random.randint(W)
    # cy = np.random.randint(H)

    # bbx1 = np.clip(cx - cut_w // 2, 0, W)
    # bby1 = np.clip(cy - cut_h // 2, 0, H)
    # bbx2 = np.clip(cx + cut_w // 2, 0, W)
    # bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    # torch.cuda.empty_cache()
    # gc.collect()
    save_dir = increment_path(os.path.join(model_dir, args.name))

    wandb.run.name = save_dir
    wandb.run.save()
    
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes   # 8

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)
    
    print('dataset len : ',len(dataset))
    
    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    print('imbalanced sampler off')
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )
    
    # print('imbalanced sampler on')
    # train_loader = DataLoader(              # 균형 샘플러 사용
    #     train_set,
    #     sampler=ImbalancedDatasetSampler(train_set),
    #     batch_size=args.batch_size,
    #     num_workers=multiprocessing.cpu_count() // 2,
    #     #shuffle=True,
    #     pin_memory=use_cuda,
    #     drop_last=True,
    # )
    

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,                           # 변경
        pin_memory=use_cuda,
        drop_last=True,
    )
    


    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy

    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    #scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    #scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=2e-5)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, patience = 5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_epoch = 0
    best_val_f1 = 0
    best_val_loss = np.inf

    for epoch in tqdm(range(args.epochs)):
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, mask, gender, age, labels  = train_batch

            inputs = inputs.to(device)
            mask = mask.to(device)
            gender = gender.to(device)
            age = age.to(device)
            labels = labels.to(device)
            
            # # # cutmix 코드
            # r = np.random.rand(1)

            # beta = 0.5
            # if beta > 0 and r < 0.5:
            #     #lam = np.random.beta(beta, beta)
            #     lam = 0.5 # 0.5 고정으로 수정
            #     rand_index = torch.randperm(inputs.size()[0]).cuda()

            #     age_a = age
            #     age_b = age[rand_index]

            #     mask_a = mask
            #     mask_b = mask[rand_index]

            #     gender_a = gender
            #     gender_b = gender[rand_index]
                
            #     bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            #     inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

            #     # adjust lambda to exactly match pixel ratio
            #     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            #     # compute output
            #     save_image(inputs[0], './test.jpg')
                
            #     mask_out, gender_out, age_out = model(inputs)

            #     mask_loss = criterion(mask_out, mask_a) * lam + criterion(mask_out, mask_b) * (1. - lam)
            #     gender_loss = criterion(gender_out, gender_a) * lam + criterion(gender_out, gender_b) * (1. - lam)
            #     age_loss = criterion(age_out, age_a) * lam + criterion(age_out, age_b) * (1. - lam)

            # else:
            #     # compute output
            #     mask_out, gender_out, age_out = model(inputs)

            #     mask_loss = criterion(mask_out, mask)
            #     gender_loss = criterion(gender_out, gender)
            #     age_loss = criterion(age_out, age)

            optimizer.zero_grad()
            
            mask_out, gender_out, age_out = model(inputs)

            mask_loss = criterion(mask_out, mask)
            gender_loss = criterion(gender_out, gender)
            age_loss = criterion(age_out, age)

            # outs = model(inputs)
            # loss = criterion(outs, labels)
            # preds = torch.argmax(outs, dim=-1)

            loss = mask_loss + gender_loss + age_loss

            mask_out = mask_out.argmax(dim=-1)
            gender_out = gender_out.argmax(dim=-1)
            age_out = age_out.argmax(dim=-1)

            preds = mask_out * 6 + gender_out * 3 + age_out

            #preds = torch.argmax(output, dim=-1)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
            

        #scheduler.step()
        
        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []

            val_labels = []  # labal,pred 추가
            val_preds = []

            figure = None
            for val_batch in val_loader:
                inputs, mask, gender, age, labels  = val_batch

                inputs = inputs.to(device)
                mask = mask.to(device)
                gender = gender.to(device)
                age = age.to(device)
                labels = labels.to(device)

                mask_out, gender_out, age_out = model(inputs)

                mask_loss = criterion(mask_out, mask)
                gender_loss = criterion(gender_out, gender)
                age_loss = criterion(age_out, age)
                
                loss = mask_loss + gender_loss + age_loss

                mask_out = mask_out.argmax(dim=-1)
                gender_out = gender_out.argmax(dim=-1)
                age_out = age_out.argmax(dim=-1)

                preds = mask_out * 6 + gender_out * 3 + age_out
                #preds = torch.argmax(outs, dim=-1)

                loss_item = loss.item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                val_labels.extend(labels.cpu().numpy()) # labal,pred 저장
                val_preds.extend(preds.cpu().numpy())

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_f1_score = metrics.f1_score(y_true=val_labels, y_pred=val_preds, average='macro') # f1_score
            #print('f1_score :', val_f1_score)
            
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_f1_score > best_val_f1:
                print(f"New best model for val F1_score : {val_f1_score:.5}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_f1 = val_f1_score
                best_epoch = epoch
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] F1_score : {val_f1_score:.5}, loss: {val_loss:4.2} || "
                f"best F1_score : {best_val_f1:.5}, best loss: {best_val_loss:4.2}, best epoch: {best_epoch}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/F1_score", val_f1_score, epoch)
            logger.add_figure("results", figure, epoch)

            scheduler.step(val_loss)

            wandb.log({"Train loss": train_loss,
                       "Train accuracy": train_acc,
                       "Val loss": val_loss,
                       "Val acc": val_acc,
                       "Val F1_Score": val_f1_score})

            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BestAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=50, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    wandb.init(project='lv1_exp')
    wandb.config.update(args)

    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
