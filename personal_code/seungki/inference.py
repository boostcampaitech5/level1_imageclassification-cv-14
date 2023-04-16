import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
import json

from dataset import TestDataset, MaskBaseDataset
from datetime import datetime



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
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())
    
    info['ans'] = preds
    
    save_path = os.path.join(output_dir, 'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    now = datetime.now()
    folder_name = now.strftime('%Y-%m-%d-%H:%M:%S')
    
    # Configuration json file
    parser.add_argument('--config', type=str, default=None, help='config file path (default: None)')

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (224, 224))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--eval_data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--bestpth_model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/'))
    # parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', f'./output/{folder_name}'))

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
        bestpth_model_dir = config['model_dir'] + '/' + config['model'] + '_' + str(config['epochs']) + '_' + str(config['batch_size']) + '_' + str(config['lr']) + '_' + str(config['augmentation']) + '_' + str(config['augmentation_types'])
        
        # -- Set output directory
        output_dir = f"./output/{folder_name}"+ config['model'] + '_' + str(config['epochs']) + '_' + str(config['batch_size']) + '_' + str(config['lr']) + '_' + str(config['augmentation']) + '_' + str(config['augmentation_types'])
        
    else:
        config = {}
        bestpth_model_dir = args.bestpth_model_dir
        output_dir = args.output_dir
    
    eval_data_dir = args.eval_data_dir
    
    os.makedirs(output_dir, exist_ok=True)

    inference(eval_data_dir, bestpth_model_dir, output_dir, args)
