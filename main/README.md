# main code

### ⭐File Tree⭐

```bash

  ├─dataset.py : dataset
  ├─inference.py : Single model inference
  ├─train.py : Single model train
  ├─loss.py : losses
  ├─model.py : models
  ├─multi_inference.py : Multi model inference
  ├─multi_train.py : Multi model train
  ├─utils.py : tools for training
  ├─voting.py : Ensemble - Hard voting
  ├─README.md 
``` 


### 모델 Train 방법
---

1. Single Model Train
``` bash
python train.py --epoch 30 --batch_size 32 --valid_batch_size 128 --model EfficientnetB4 --resize 380 380 --criterion focal
```
- `argparse`는 사용자의 선택에 따라서 수정 가능합니다. `argparse` 종류는 `train.py`코드에서 확인해주세요

---

2. Multi Model Train
``` bash
python multi_train.py --epoch 30 --batch_size 32 --valid_batch_size 128 --model EfficientnetB43way --resize 380 380 --criterion focal
```
- `argparse`는 사용자의 선택에 따라서 수정 가능합니다. `argparse` 종류는 `multi_train.py`코드에서 확인해주세요


### 모델 Inference 방법
---

1. Single Model Inference 
```bash
python inference.py --model EfficientnetB4 --batch_size 256 --model_dir ./model/exp
```
- `--model_dir`의 경우 사용자의 모델이 저장된 경로로 지정해야합니다
- `argparse`는 사용자의 선택에 따라서 수정 가능합니다. `argparse` 종류는 `inference.py`코드에서 확인해주세요

---

2. Multi Model Inference 
```bash
python multi_inference.py --model EfficientnetB43wayF --batch_size 256 --model_dir ./model/exp
```
- `--model_dir`의 경우 사용자의 모델이 저장된 경로로 지정해야합니다
- `argparse`는 사용자의 선택에 따라서 수정 가능합니다. `argparse` 종류는 `multi_inference.py`코드에서 확인해주세요


### Ensemble 방법
---

- `voting.py`에서 `submit csv`의 경로를 `submit_csv`변수에 지정해주고, **Ensemble**할 `model.pth`들이 모여있는 폴더의 경로를 (하나의 폴더에 모아줍니다) `path`에 지정

```bash
python voting
```
- `voting`파이썬을 실행해주면 `path`위치에 앙상블의 결과가 `voting.csv`로 저장됩니다


### TOP1 Model 학습 방법
---

```bash
A : Single Model (EfficientnetB4) + 57~59 Remove + Mix-up data
B : Single Model (EfficientnetB4) + 57~59 Remove + Custom Cutmix
C : pretrained A weight and Multi Model (EfficientnetB43wayF) + 57~59 Remove + rembg & deepface 

and 

- A + B + C : Hard voting
```

**Public Score**
![public](../Image/public.png)

**Private Score**
![private](../Image/private.png)

