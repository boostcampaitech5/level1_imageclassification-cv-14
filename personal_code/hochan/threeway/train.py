import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime
from lion_pytorch.lion_pytorch import Lion

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from torchvision.transforms import Compose
from torchvision import transforms
from torchvision.utils import save_image

from dataset import MaskBaseDataset, MaskSplitByProfileDataset
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter, RandomErasing, RandomHorizontalFlip, RandomRotation
from loss import create_criterion
from dataset import rand_bbox
from dataset import FaceNet
import wandb

import gc
gc.collect()
torch.cuda.empty_cache()


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


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, f"{args.model}_{args.epochs}_{args.batch_size}_{args.lr}_{args.criterion}_{args.optimizer}"))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
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
    criterion = create_criterion(args.criterion) 

    if args.optimizer == 'Lion':
        optimizer = Lion(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4)
        
    else:
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )

    scheduler =  ReduceLROnPlateau(optimizer, patience=5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    train_transform = Compose([
            RandomErasing(p=0.5),
            RandomHorizontalFlip(p=0.5),
    ])

    val_transform = Compose([
            Resize(args.resize, Image.BILINEAR),
    ])


    best_f1 = 0

    for epoch in range(args.epochs):
        # train loop
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

            optimizer.zero_grad()

            Cut_Prob = np.random.rand(1)

            if args.cutmix and Cut_Prob > args.cutmix_prob:
                lam = np.random.beta(0.5, 0.5)
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                
                age_a = age
                age_b = age[rand_index]

                mask_a = mask
                mask_b = mask[rand_index]

                gender_a = gender
                gender_b = gender[rand_index]

                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

                #inputs = train_transform(inputs)

                #check 
                #save_image(inputs[0], f'./cutmix_image/{idx}_cutmix.png')

                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

                mask_out, gender_out, age_out = model(inputs)

                mask_loss = criterion(mask_out, mask_a) * lam + criterion(mask_out, mask_b) * (1. - lam)
                gender_loss = criterion(gender_out, gender_a) * lam + criterion(gender_out, gender_b) * (1. - lam)
                age_loss = criterion(age_out, age_a) * lam + criterion(age_out, age_b) * (1. - lam)

            else:
                inputs = train_transform(inputs)

                mask_out, gender_out, age_out = model(inputs)

                mask_loss = criterion(mask_out, mask)
                gender_loss = criterion(gender_out, gender)
                age_loss = criterion(age_out, age)

            loss = mask_loss + gender_loss + age_loss
            loss.backward()

            optimizer.step()
            mask_out = mask_out.argmax(dim=-1)
            gender_out = gender_out.argmax(dim=-1)
            age_out = age_out.argmax(dim=-1)

            preds = mask_out * 6 + gender_out * 3 + age_out
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
                wandb.log({"train_loss": train_loss}, step=epoch)
                loss_value = 0
                matches = 0

        
        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = 1

            y_true = []
            y_pred = []

            for val_batch in val_loader:
                inputs, mask, gender, age, labels  = val_batch
                inputs = inputs.to(device)
                mask = mask.to(device)
                gender = gender.to(device)
                age = age.to(device)
                labels = labels.to(device)

                inputs = val_transform(inputs)

                mask_out, gender_out, age_out = model(inputs)
                
                mask_loss = criterion(mask_out, mask)
                gender_loss = criterion(gender_out, gender)
                age_loss = criterion(age_out, age)

                loss = mask_loss + gender_loss + age_loss

                mask_out = mask_out.argmax(dim=-1)
                gender_out = gender_out.argmax(dim=-1)
                age_out = age_out.argmax(dim=-1)
            
                #통합라벨로 변환
                preds = mask_out * 6 + gender_out * 3 + age_out

                loss_item = loss.item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                y_true += labels.cpu().tolist()
                y_pred += preds.cpu().tolist()

                

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )
            
            f1 = f1_score(y_true, y_pred, average='macro')
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)

            wandb.log({"val_loss": val_loss, "val_acc": val_acc, 'val_f1': f1}, step=epoch)
            if f1 > best_f1:
                print(f"New best model for F1 : {f1:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_f1 = f1
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best F1 : {best_f1:4.2%} ||"
            )
            print(f"[F1] : {f1:4.2}")
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            scheduler.step(val_loss)
            print()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[380, 380], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--valid_batch_size', type=int, default=128, help='input batch size for validing (default: 128)')
    parser.add_argument('--model', type=str, default='EfficientB43way', help='model type (default: EfficientB43way)')
    parser.add_argument('--optimizer', type=str, default='Adam', help=f'optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default=f'focal', help=f'criterion type (default: focal)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=50, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--cutmix', default=False, help='Do you use CutMix T/F (default : False)')
    parser.add_argument('--cutmix_prob', type=float, default=0.5, help = 'Cut mix pro setting (0~1, type=float, default:0.5)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir

    wandb.init(project=args.model)

    config={
    "learning_rate": args.lr,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "optimizer": args.optimizer,
    "model": args.model,
    "seed": args.seed,
    "criterion": args.criterion,

    }
    
    wandb.config.update(config)

    train(data_dir, model_dir, args)
