import numpy as np
import torch

#rand_bbox for Single Model Cutmix, Multi Model Cutmix
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]

    bbx1 = np.clip(0, 0, W)
    bby1 = np.clip(H // 2, 0, H)
    bbx2 = np.clip(W, 0, W)
    bby2 = np.clip(H, 0, H)

    return bbx1, bby1, bbx2, bby2


#Single Model Cutmix
def cutmix(labels, inputs, model, criterion):
    # 0.5 고정으로 수정
    lam = 0.5  

    rand_index = torch.randperm(inputs.size()[0]).cuda()
    target_a = labels
    target_b = labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

    output = model(inputs)
    loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1.0 - lam)

    return output, loss


#Multi Model Cutmix
def multi_label_cutmix(age, mask, gender, inputs, model, criterion):
    # 0.5 고정으로 수정
    lam = 0.5  

    rand_index = torch.randperm(inputs.size()[0]).cuda()

    age_a = age
    age_b = age[rand_index]

    mask_a = mask
    mask_b = mask[rand_index]

    gender_a = gender
    gender_b = gender[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

    # compute output
    mask_out, gender_out, age_out = model(inputs)

    mask_loss = criterion(mask_out, mask_a) * lam + criterion(mask_out, mask_b) * (
        1.0 - lam
    )
    gender_loss = criterion(gender_out, gender_a) * lam + criterion(
        gender_out, gender_b
    ) * (1.0 - lam)
    age_loss = criterion(age_out, age_a) * lam + criterion(age_out, age_b) * (1.0 - lam)

    return mask_out, gender_out, age_out, mask_loss, gender_loss, age_loss

