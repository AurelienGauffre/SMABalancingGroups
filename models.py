# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torchvision
from transformers import BertForSequenceClassification, AdamW, get_scheduler
from pytorch_metric_learning import losses
import os
class ToyNet(torch.nn.Module):
    def __init__(self, dim, gammas):
        super(ToyNet, self).__init__()
        # gammas is a list of three the first dimension determines how fast the
        # spurious feature is learned the second dimension determines how fast
        # the core feature is learned and the third dimension determines how
        # fast the noise features are learned
        self.register_buffer(
            "gammas", torch.tensor([gammas[:2] + gammas[2:] * (dim - 2)])
        )
        self.fc = torch.nn.Linear(dim, 1, bias=False)
        self.fc.weight.data = 0.01 / self.gammas * self.fc.weight.data

    def forward(self, x):
        return self.fc((x * self.gammas).float()).squeeze()


class BertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(
            input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            token_type_ids=x[:, :, 2]).logits


def get_bert_optim(network, lr, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    nodecay_params = []
    for n, p in network.named_parameters():
        if any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8)
    return optimizer


def get_sgd_optim(network, lr, weight_decay):
    return torch.optim.SGD(
        network.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9)




class CustomResNet(torch.nn.Module):
    # On a besoin de définir un nouveau modèle car on a besoin de changer la dernière couche pour le contrastive learning qui a besoin de l'embedddings
    def __init__(self, arch, n_classes,pretrained_path=None):
        super(CustomResNet, self).__init__()
        self.n_classes = n_classes

        # Load the pre-trained ResNet model
        if arch == "resnet18":
            self.network = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        elif arch == "resnet50":
            self.network = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            
        if pretrained_path is not None:
            print(f"############## Loading pretrained model {pretrained_path} ##############")
            self.network.load_state_dict(torch.load(os.path.join('checkopint',pretrained_path)))
            
        self.feature_extractor = torch.nn.Sequential(*list(self.network.children())[:-1])
        
        # Replace the final fully connected layer to match the number of classes
        self.fc = torch.nn.Linear(self.network.fc.in_features, self.n_classes)

    def forward(self, x):
        # Extract features (embeddings)
        embeddings = self.feature_extractor(x)
        embeddings = embeddings.view(embeddings.size(0), -1)  # Flatten the embeddings
        
        # Calculate the logits using the new fully connected layer
        logits = self.fc(embeddings)
        # We return the embeddings and the logits to use both supcon and bce losses
        return embeddings, logits
    
    



class ERM(torch.nn.Module):
    def __init__(self, hparams, dataloader):
        super().__init__()
        self.hparams = dict(hparams)
        self.cl_mode = self.hparams.get('cl_mode', 'bce') #contrastive learning mode (default bce, no contrastive)
        dataset = dataloader.dataset
        self.n_batches = len(dataloader)
        self.data_type = dataset.data_type
        self.n_classes = len(set(dataset.y))
        self.n_groups = len(set(dataset.g))
        self.n_examples = len(dataset)
        self.last_epoch = 0
        self.best_selec_val = 0
        self.pretrained_path = self.hparams.get('pretrained_path', None)
        
        self.init_model_(self.data_type, text_optim="sgd", arch=self.hparams['arch'])

    def init_model_(self, data_type, text_optim="sgd", arch="resnet18"):
        self.clip_grad = text_optim == "adamw"
        optimizers = {
            "adamw": get_bert_optim,
            "sgd": get_sgd_optim
        }

        if data_type == "images":
            # old way to call the networks
            # # if arch == "resnet18":

            #     self.network = torchvision.models.resnet.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            # elif arch == "resnet50":
            #     self.network = torchvision.models.resnet.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            
            # self.network.fc = torch.nn.Linear(self.network.fc.in_features, self.n_classes)
            
            self.network = CustomResNet(arch=arch, n_classes=self.n_classes, pretrained_path=self.pretrained_path)


            self.optimizer = optimizers['sgd'](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay'])

            if self.hparams.get('scheduler', False):
                print("Using cosine annealing scheduler")
                num_training_steps = int(self.hparams["num_epochs"]) * self.n_batches
                # Assuming T_max and eta_min as hyperparameters
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=num_training_steps, eta_min=0)
            else:
                self.lr_scheduler = None
            # if the dictionnary does not contain the key "cl_mode" or contains the key with value "classic"
           
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")
            
            self.loss_cl = losses.SupConLoss(temperature=0.07) 

        # elif data_type == "text":
        #     self.network = BertWrapper(
        #         BertForSequenceClassification.from_pretrained(
        #             'bert-base-uncased', num_labels=self.n_classes))
        #     self.network.zero_grad()
        #     self.optimizer = optimizers[text_optim](
        #         self.network,
        #         self.hparams['lr'],
        #         self.hparams['weight_decay'])

        #     num_training_steps = self.hparams["num_epochs"] * self.n_batches
        #     self.lr_scheduler = get_scheduler(
        #         "linear",
        #         optimizer=self.optimizer,
        #         num_warmup_steps=0,
        #         num_training_steps=num_training_steps)
        #     self.loss = torch.nn.CrossEntropyLoss(reduction="none")

        # elif data_type == "toy":
        #     gammas = (
        #         self.hparams['gamma_spu'],
        #         self.hparams['gamma_core'],
        #         self.hparams['gamma_noise'])

        #     self.network = ToyNet(self.hparams['dim_noise'] + 2, gammas)
        #     self.optimizer = optimizers['sgd'](
        #         self.network,
        #         self.hparams['lr'],
        #         self.hparams['weight_decay'])
        #     self.lr_scheduler = None
        #     self.loss = lambda x, y: \
        #         torch.nn.BCEWithLogitsLoss(reduction="none")(x.squeeze(),
        #                                                      y.float())

        self.cuda()

    def compute_loss_value_(self, i, x, y, g, epoch):
        if self.cl_mode == 'bce':
            return self.loss(self.network(x)[1], y).mean()
        elif self.cl_mode == 'bce+supcon':
            # print bce and supcon losses values :
            print("BCE/CL: ")
            return self.loss(self.network(x)[1], y).mean() * (1-self.hparams['alpha']) + self.loss_cl(self.network(x)[0], y).mean() * self.hparams['alpha']

    def update(self, i, x, y, g, epoch):
        x, y, g = x.cuda(), y.cuda(), g.cuda()
        loss_value = self.compute_loss_value_(i, x, y, g, epoch)

        if loss_value is not None:
            self.optimizer.zero_grad()
            loss_value.backward()

            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)

            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.data_type == "text":
                self.network.zero_grad()

            loss_value = loss_value.item()

        self.last_epoch = epoch
        return loss_value

    def predict(self, x):
        return self.network(x)[1] #self.network(x)[1] renvoie maintenant les logits

    def accuracy(self, loader):
        nb_groups = loader.dataset.nb_groups
        nb_labels = loader.dataset.nb_labels
        corrects = torch.zeros(nb_groups * nb_labels)
        totals = torch.zeros(nb_groups * nb_labels)
        self.eval()
        with torch.no_grad():
            for i, x, y, g in loader:
                predictions = self.predict(x.cuda())
                if predictions.squeeze().ndim == 1:
                    predictions = (predictions > 0).cpu().eq(y).float()
                else:
                    predictions = predictions.argmax(1).cpu().eq(y).float()
                groups = (nb_groups * y + g)
                for gi in groups.unique():
                    corrects[gi] += predictions[groups == gi].sum()
                    totals[gi] += (groups == gi).sum()
        corrects, totals = corrects.tolist(), totals.tolist()
        self.train()
        return sum(corrects) / sum(totals), \
            [c / t if t != 0 else 0 for c, t in zip(corrects, totals)]

    def load(self, fname):
        dicts = torch.load(fname)
        self.last_epoch = dicts["epoch"]
        self.load_state_dict(dicts["model"])
        self.optimizer.load_state_dict(dicts["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(dicts["scheduler"])

    def save(self, fname):
        lr_dict = None
        if self.lr_scheduler is not None:
            lr_dict = self.lr_scheduler.state_dict()
        torch.save(
            {
                "model": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": lr_dict,
                "epoch": self.last_epoch,
                "best_selec_val": self.best_selec_val,
            },
            fname,
        )


class GroupDRO(ERM):
    def __init__(self, hparams, dataset):
        super(GroupDRO, self).__init__(hparams, dataset)
        self.register_buffer(
            "q", torch.ones(self.n_classes * self.n_groups).cuda())

    def groups_(self, y, g):
        idx_g, idx_b = [], []
        all_g = y * self.n_groups + g

        for g in all_g.unique():
            idx_g.append(g)
            idx_b.append(all_g == g)

        return zip(idx_g, idx_b)

    def compute_loss_value_(self, i, x, y, g, epoch):
        losses = self.loss(self.network(x), y)

        for idx_g, idx_b in self.groups_(y, g):
            self.q[idx_g] *= (
                    self.hparams["eta"] * losses[idx_b].mean()).exp().item()

        self.q /= self.q.sum()

        loss_value = 0
        for idx_g, idx_b in self.groups_(y, g):
            loss_value += self.q[idx_g] * losses[idx_b].mean()

        return loss_value


class JTT(ERM):
    def __init__(self, hparams, dataset):
        super(JTT, self).__init__(hparams, dataset)
        self.register_buffer(
            "weights", torch.ones(self.n_examples, dtype=torch.long).cuda())

    def compute_loss_value_(self, i, x, y, g, epoch):
        if epoch == self.hparams["T"] + 1 and \
                self.last_epoch == self.hparams["T"]:
            self.init_model_(self.data_type, text_optim="adamw", arch=self.hparams['arch'])

        predictions = self.network(x)

        if epoch != self.hparams["T"]:
            loss_value = self.loss(predictions, y).mean()
        else:
            self.eval()
            if predictions.squeeze().ndim == 1:
                wrong_predictions = (predictions > 0).cuda().ne(y.cuda()).float()  # TODO Added .cuda() to fix error
            else:
                wrong_predictions = predictions.argmax(1).cuda().ne(y.cuda()).float()  # TODO Added .cuda() to fix error

            # print("DEBUG SHAPE Weight[i]: ", self.weights[i].shape)
            print(self.weights[i])
            # print("DEBUG SHAPE wrong_predictions: ", wrong_predictions.shape)

            if self.weights[i].shape == wrong_predictions.shape:
                self.weights[i] += (
                        wrong_predictions.detach() * (self.hparams["up"] - 1)).long()  # TODO Added .long() to fix error
            else:
                print("Dimension error, adding zeros of the same size as self.weights[i] tofix")
                # Add zeros of the same size as self.weights[i]
                self.weights[i] += torch.zeros_like(self.weights[i]).long()
            self.train()
            loss_value = None

        return loss_value

    def load(self, fname):
        dicts = torch.load(fname)
        self.last_epoch = dicts["epoch"]

        if self.last_epoch > self.hparams["T"]:
            self.init_model_(self.data_type, text_optim="adamw", arch=self.hpams['arch'])

        self.load_state_dict(dicts["model"])
        self.optimizer.load_state_dict(dicts["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(dicts["scheduler"])
