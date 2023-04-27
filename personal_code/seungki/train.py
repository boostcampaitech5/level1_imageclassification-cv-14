import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configparser import ConfigParser
from dataset import MaskBaseDataset
from loss import create_criterion

import wandb

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import gc

#-- warning ignored
# warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', message="A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy")

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


# def grid_image(np_images, gts, preds, n=16, shuffle=False):
#     batch_size = np_images.shape[0]
#     assert n <= batch_size

#     choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
#     figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
#     plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
#     n_grid = int(np.ceil(n ** 0.5))
#     tasks = ["mask", "gender", "age"]
#     for idx, choice in enumerate(choices):
#         gt = gts[choice].item()
#         pred = preds[choice].item()
#         image = np_images[choice]
#         gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
#         pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
#         title = "\n".join([
#             f"{task} - gt: {gt_label}, pred: {pred_label}"
#             for gt_label, pred_label, task
#             in zip(gt_decoded_labels, pred_decoded_labels, tasks)
#         ])

#         plt.subplot(n_grid, n_grid, idx + 1, title=title)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(image, cmap=plt.cm.binary)

#     return figure


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
        return f"{path}({n})"


def train(data_dir, model_dir, args):
    

    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, f"{args.model}_{args.epochs}_{args.batch_size}_{args.lr}_{args.augmentation}_{args.criterion}"))

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
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.1)
#     scheduler = CosineAnnealingLR(optimizer, args.lr_decay_step)
#     scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=args.lr_decay_step)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, mode="min", patience=args.lr_decay_step)
    
    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, f'{args.model}_{args.epochs}_{args.batch_size}_{args.lr}_config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_f1_score = 0
    best_val_loss = np.inf
    
    # -- train loop
    
    earlystop_cnt = 0
    
    for epoch in range(args.epochs):
        # torch.cuda.empty_cache()
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()
            
            
            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"EPOCH[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"TRAINING LOSS: {train_loss:4.4} || TRAINING ACC: {train_acc:4.2%} || LR: {current_lr}"
                )
                # logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                # logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                
                # -- wandb 학습 단계에서 Loss, Accuracy 로그 저장
                wandb.log({
                "Train loss": train_loss,
                "Train acc" : train_acc
            })
                
                
                loss_value = 0
                matches = 0

            # del inputs
            # del labels
            
        # -- steplr scheduler step    
        # scheduler.step()

        # -- val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            
            # -- answer label & prediction
            y_true = []
            y_pred = []
            
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                
                # -- append to list
                y_true += labels.cpu().tolist()
                y_pred += preds.cpu().tolist()
            
                
                # if figure is None:
                #     inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                #     inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                #     figure = grid_image(
                #         inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                #     )
            
            # auroc = roc_auc_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average="macro")
            # best_f1_score = max(best_f1_score, f1)
            precision = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")    
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            
            # -- scheduler step for plateu min
            scheduler.step(val_loss)
            
            # -- Early Stopping, 
            # if (val_loss > best_val_loss) or (val_acc < best_val_acc):
            #     earlystop_cnt+=1

            #     if earlystop_cnt == 17:
            #         print("EARLY STOPPED. NO SIGNIFICANT CHANGE IN VALIDATION PERFORMANCE FOR 17 EPOCHS")
            #         break
            # else:
            #     earlystop_cnt = 0
            
            # -- best val acc save
            if val_acc > best_val_acc:
#                 print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
#                 torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
#             torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            
            # -- best f1 score save
            if f1 > best_f1_score:
                print(f"New best model for f1 score : {f1:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_f1_score = f1
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            
            
            print(
                f"[VAL] ACC : {val_acc:4.2%}, LOSS : {val_loss:4.2} || "
                f"BEST ACC : {best_val_acc:4.2%}, BEST LOSS : {best_val_loss:4.2} || "
                f"F1 SCORE : {f1:4.2%}, BEST F1 SCORE : {best_f1_score:4.2%} ||"
                # f"AUROC : {auroc:4.2%}"
                # f"precision : {precision:4.2%}, recall : {recall:4.2%}"
            )
            # logger.add_scalar("Val/loss", val_loss, epoch)
            # logger.add_scalar("Val/accuracy", val_acc, epoch)
            # logger.add_figure("results", figure, epoch)
            
            # -- wandb 검증 단계에서 Loss, Accuracy 로그 저장
            wandb.log({
                "VALID LOSS": val_loss,
                "VALID ACC" : val_acc,
                "F1 SCORE" : f1,
                "BEST VALID ACC" : best_val_acc
            })
            
            print()
            
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #-- Configuration json file
    parser.add_argument('--config', type=str, default=None, help='config file path (default: None)')

    #-- Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    
    #-- parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    
    #-- parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: CustomAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=256, help='input batch size for validing (default: 256)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate scheduler deacy step (default: 5)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default=None, help='model save at {SM_MODEL_DIR}/{name}')
    
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau', help='scheduler')
    
    #-- pass on arguments for wandb
    parser.add_argument('--augmentation_types', default=None, help='augmentation logging for wandb')
    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    
    # -- mixup train
    # parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input_mixup/data/train/images'))
    
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    
    if args.name is None:
        args.name = args.model
    
    # -- Read configuration values    
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # -- Override default configuration with configuration from config file
        for key in config:
            if key in args.__dict__ and config[key] is not None:
                args.__dict__[key] = config[key]
        
        # -- wandb runname using augmentation types
#         wandb_runname = config['model'] + '_' + str(config['epochs']) + '_' + str(config['batch_size']) + '_' + str(config['lr']) + '_' + str(config['augmentation']) + '_' + str(config['augmentation_types'])
        
        # -- wandb runname using name of augmentation class
        wandb_runname = config['model'] + '_' + str(config['epochs']) + '_' + str(config['batch_size']) + '_' + str(config['lr']) + '_' + str(config['augmentation'])
        
    else:
        config = {}
    
    
    # -- wandb configuration
    
    # 1. name with augmentation types(or notes)
    # wandb_runname = f'{args.model}_{args.batch_size}_{args.lr}_{args.augmentation}_{args.augmentation_types}'
    
    # 2. name with without augmentation types(or notes)
    wandb_runname = f'{args.model}_{args.batch_size}_{args.lr}_{args.augmentation}'
    
    # 3. name for hyperparameter
    # wandb_runname = f'{args.model}_{args.batch_size}_{args.lr}_{args.augmentation}_{args.criterion}_{args.scheduler}_{args.optimizer}_{args.weight_decay}_{args.lr_decay_step}'
    

    # project_name = "Image Classification Competition for Naver Boostcamp AI Tech"
    # project_name = "Custom augmentation experiments"

    # project_name = "Augmentation comparison one by one"
    
    # project_name = "Final Models"
    project_name = "Augmentation without earlystopping"
    # project_name = "Test run 1"
    wandb.init(project=project_name,name=wandb_runname)
    
    wandb_config={
    "model": args.model,
    "learning_rate": args.lr,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "optimizer": args.optimizer,
    "seed": args.seed,
    "criterion": args.criterion,
    "resize": args.resize,
    "augmentation" : args.augmentation,
    "lr_decay_step" : args.lr_decay_step,
    "weight_decay" : args.weight_decay,
    "scheduler" : args.scheduler,
    "augmentation_types" : args.augmentation_types
    }
    
    wandb.config.update(wandb_config)
        
    
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)