import argparse
import multiprocessing
import os
from importlib import import_module
import numpy as np
import random
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.metrics import classification_report , confusion_matrix, ConfusionMatrixDisplay

from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def evaluation(data_dir, model_dir, args):
    """
    """
    seed_everything(args.seed)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    #model.eval()

    # img_root = os.path.join(data_dir, 'images')
    # info_path = os.path.join(data_dir, 'info.csv')
    # info = pd.read_csv(info_path)
    # img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

	# dataset = TestDataset(img_paths, args.resize)

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
    
    train_set, eval_set = dataset.split_dataset()

    eval_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    with torch.no_grad():
        print("Calculating evaluation results...")
        model.eval()
        
        preds = []
        real_label = []

        for eval_batch in eval_loader:
            
            inputs, labels = eval_batch
            inputs = inputs.to(device)
            #labels = labels.to(device)

            outs = model(inputs)
            
            model_pred = torch.argmax(outs, dim=-1)
            
            #print(type(preds),type(labels))
            
            # loss_item = criterion(outs, labels).item()
            # acc_item = (labels == preds).sum().item()
            # val_loss_items.append(loss_item)
            # val_acc_items.append(acc_item)

            preds.extend(model_pred.cpu().numpy())
            real_label.extend(labels.cpu().numpy())
            
            #print(type(preds),type(labels))
            
            #print(len(preds),len(real_label))

        #print(type(preds),type(real_label))
        #print(preds)
        #print(real_label)
    
        print(classification_report(real_label, preds))
        print(confusion_matrix(real_label, preds))
    
    # print("Calculating inference results..")
    # preds = []
    # with torch.no_grad():
    #     for idx, images in enumerate(eval_loader):
    #         images = images.to(device)
    #         pred = model(images)
    #         pred = pred.argmax(dim=-1)
    #         preds.extend(pred.cpu().numpy())

    # info['ans'] = preds
    # save_path = os.path.join(output_dir, f'output.csv')
    # info.to_csv(save_path, index=False)
    # print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Configuration json file
    parser.add_argument('--config', type=str, default=None, help='config file path (default: None)')
    
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    # parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    
    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/train/images'))
    parser.add_argument('--bestpth_model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/'))
    #parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()
    
    # -- Read configuration values    
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # -- Override default configuration with configuration from config file
        for key in config:
            if key in args.__dict__ and config[key] is not None:
                args.__dict__[key] = config[key]
        
        # -- Set best.pth path
        bestpth_model_dir = config['model_dir'] + '/' + config['model'] + '_' + str(config['epochs']) + '_' + str(config['batch_size']) + '_' + str(config['lr'])
        
    else:
        config = {}
        bestpth_model_dir = args.bestpth_model_dir
        
    
    data_dir = args.data_dir


    evaluation(data_dir, bestpth_model_dir, args)
