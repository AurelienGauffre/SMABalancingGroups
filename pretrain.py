import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightly
import wandb
from omegaconf import OmegaConf
from datasets import get_loaders
import models
from utils import Tee, flatten_dictionary_for_wandb
from torchvision.transforms import ToPILImage
from lightly.models.simsiam import SimSiam
import torchvision
import copy
from lightly.loss import NegativeCosineSimilarity, NTXentLoss
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead, MoCoProjectionHead
from lightly.transforms import SimSiamTransform, MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import deactivate_requires_grad, update_momentum

def parse_args():
    parser = argparse.ArgumentParser(description='Pretraining configurations.')
    parser.add_argument('--config', type=str, default='configZ1.yaml')
    return parser.parse_args()

class SimSiam(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(2048, 2048, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

class MoCo(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(2048, 2048, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

def main():
    os.environ["WANDB__SERVICE_WAIT"] = "500"

    args = parse_args()
    config = OmegaConf.load(os.path.join('configs', args.config))

    wandb.init(project=config.wandb_project, config=flatten_dictionary_for_wandb(dict(config)))

    resnet = torchvision.models.resnet50()  # TO ADAPT
    backbone = nn.Sequential(*list(resnet.children())[:-1])

    # Select model based on config
    if config.loss == 'simsiam':
        model = SimSiam(backbone)
        transform = SimSiamTransform(input_size=config.SMA.img_size)
        criterion = NegativeCosineSimilarity()
    elif config.loss == 'moco':
        model = MoCo(backbone)
        transform = MoCoV2Transform(input_size=config.SMA.img_size)
        criterion = NTXentLoss(memory_bank_size=(4096, 128))
    else:
        raise ValueError(f"Unsupported loss type: {config.loss}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

    loaders = get_loaders(config.data_path, config.dataset, config.batch_size, config.method, config.SMA, transform=transform)
    train_loader = loaders["tr"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    for epoch in range(100):
        total_loss = 0
        momentum_val = cosine_schedule(epoch, 100, 0.996, 1)  # For MoCo

        for i,x, y, g in train_loader:
            
            if config.loss == 'simsiam':
                x0, x1 = x
                x0 = x0.to(device)
                x1 = x1.to(device)
                z0, p0 = model(x0)
                z1, p1 = model(x1)
                loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
            elif config.loss == 'moco':
                x_query, x_key = x
                x_query = x_query.to(device)
                x_key = x_key.to(device)
                update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
                update_momentum(model.projection_head, model.projection_head_momentum, m=momentum_val)
                query = model(x_query)
                key = model.forward_momentum(x_key)
                loss = criterion(query, key)

            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        # Log in wandb
        wandb.log({"loss": avg_loss})
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoint/{config.loss}_{epoch+1}.ckpt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    main()
