import numpy as np
import os
from os.path import join
import json
import yaml
import random
import copy
from PIL import Image
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from network.network_builder import get_gazelle_model
from network.network_builder_update2 import get_gt360_model, get_gazemoe_model
import network.utils as utils
from train_gazefollow import GazeDataset, collate_fn
from eval import vat_auc, vat_l2


def load_data_vat(file, sample_rate):
    sequences = json.load(open(file, "r"))
    data = []
    for i in range(len(sequences)):
        for j in range(0, len(sequences[i]['frames']), sample_rate):
            data.append(sequences[i]['frames'][j])
    return data


def load_data_gazefollow(file):
    data = json.load(open(file, "r"))
    return data


def main():
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)
    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    # enforce model_name for VAT with INOUT branch!
    config['model']['name'] = "gazelle_dinov2_vitl14_inout"

    wandb.init(
        project="gazelle",
        name="train_videoattentiontarget",
        config=config
    )

    checkpoint_dir = "_".join([
        config['train']['optimizer'],
        "bs" + str(config['train']['batch_size']),
        str(config['train']['lr']),
        str(config['train']['inout_lr']),
        config['model']['pbce_loss'],
        str(config['model']['bce_weight']),
        str(config['model']['mse_weight']),
        str(config['model']['angle_weight']),
        str(config['model']['vec_weight'])
    ])
    exp_dir = os.path.join(config['logging']['log_dir'], checkpoint_dir)
    os.makedirs(exp_dir, exist_ok=True)

    # load pretrained model
    model, transform = get_gazelle_model(config)
    print("Initializing from {}".format(config['model']['pretrained_path']))

    # initializing from ckpt without inout head
    model.load_gazelle_state_dict(torch.load(config['model']['pretrained_path'], weights_only=True))
    ### if loading model incl. backbone ###
    #model.load_gazelle_state_dict(
    #    torch.load(config['model']['pretrained_path'], weights_only=True, map_location=device), include_backbone=False)
    model.to(device)

    for param in model.backbone.parameters():  # freeze backbone
        param.requires_grad = False
    print(f"Learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_dataset = GazeDataset('videoattentiontarget', config['data']['train_path'], 'train', transform,
                                in_frame_only=False, sample_rate=6)
    train_dl = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=config['train']['batch_size'],
                                           shuffle=True,
                                           collate_fn=collate_fn,
                                           num_workers=3)
    # Note this eval dataloader samples frames sparsely for efficiency - for final results, run eval_vat.py which uses sample rate 1
    eval_dataset = GazeDataset('videoattentiontarget', config['data']['test_path'], 'test', transform,
                               in_frame_only=False, sample_rate=6)  # sample_rate=6 / 1
    eval_dl = torch.utils.data.DataLoader(eval_dataset,
                                          batch_size=config['train']['batch_size'],
                                          shuffle=False,
                                          collate_fn=collate_fn,
                                          num_workers=3)

    heatmap_loss_fn = nn.BCELoss()
    inout_loss_fn = nn.BCELoss()

    param_groups = [
        {'params': [param for name, param in model.named_parameters() if "inout" in name], 'lr': config['train']['lr']},
        {'params': [param for name, param in model.named_parameters() if "inout" not in name], 'lr': config['train']['inout_lr']}
    ]
    optimizer = torch.optim.Adam(param_groups)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-7)

    for epoch in range(config['train']['epochs']):
        # TRAIN EPOCH
        model.train()
        for cur_iter, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            preds = model({"images": imgs.to(device), "bboxes": [[bbox] for bbox in bboxes]})
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)

            # compute heatmap loss only for in-frame gaze targets
            heatmap_loss = heatmap_loss_fn(heatmap_preds[inout.bool()], heatmaps[inout.bool()].to(device))
            inout_loss = inout_loss_fn(inout_preds, inout.float().to(device))

            loss = heatmap_loss + config['model']['bce_weight'] * inout_loss
            loss.backward()
            optimizer.step()

            if cur_iter % config['logging']['save_every'] == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/heatmap_loss": heatmap_loss.item(),
                    "train/inout_loss": inout_loss.item()
                })
                #print("TRAIN EPOCH {}, iter {}/{}, loss={}".format(epoch, cur_iter, len(train_dl), round(loss.item(), 4)))

        scheduler.step()

        ckpt_path = os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch))
        torch.save(model.get_gazelle_state_dict(), ckpt_path)
        print("Saved checkpoint to {}".format(ckpt_path))

        # EVAL
        model.eval()
        l2s = []
        aucs = []
        all_inout_preds = []
        all_inout_gts = []

        for cur_iter, batch in tqdm(enumerate(eval_dl), total=len(eval_dl)):
            imgs, bboxes, gazex, gazey, inout, heights, widths = batch

            with torch.no_grad():
                preds = model({"images": imgs.to(device), "bboxes": [[bbox] for bbox in bboxes]})

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)
            for i in range(heatmap_preds.shape[0]):
                if inout[i] == 1: # in-frame
                    auc = vat_auc(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    l2 = vat_l2(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    aucs.append(auc)
                    l2s.append(l2)
                all_inout_preds.append(inout_preds[i].item())
                all_inout_gts.append(inout[i])

        epoch_l2 = np.mean(l2s)
        epoch_auc = np.mean(aucs)
        epoch_inout_ap = average_precision_score(all_inout_gts, all_inout_preds)

        wandb.log({"eval/auc": epoch_auc, "eval/l2": epoch_l2, "eval/inout_ap": epoch_inout_ap, "epoch": epoch})
        print("EVAL EPOCH {}: AUC={}, L2={}, Inout AP={}".format(epoch,
                                                                 round(float(epoch_auc), 4),
                                                                 round(float(epoch_l2), 4),
                                                                 round(float(epoch_inout_ap), 4)))


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
