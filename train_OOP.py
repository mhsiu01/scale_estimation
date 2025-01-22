# Utilities
import os
import json
# import argparse
import sys
import pdb
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')

# Import training libraries: torch, tqdm, tensorboard
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
# Distributed training with torch's DDP and torchrun
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Tracking metrics + fix RNG seed for reproducibility
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from time import time
import random
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

from datasets import partitionCatalogPerItem, partitionSpinsPerObject
# from utils_OOP import _get_img_normalize_values, _get_bbox_normalize_values


def _ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return


class TrainingRun():
    def __init__(self, hparams, run_info):
        super().__init__()
        self.SEED = 123
        self.hparams = hparams
        self.run_info = run_info
        self.use_ddp = run_info['use_ddp']
        self.gpus_per_worker = run_info['gpus_per_worker']
        if self.use_ddp:
            _ddp_setup()
            self.device = "cuda:" + str(os.environ["LOCAL_RANK"])
        else:
            self.device = "cuda"
        self.on_log_device = True if (self.use_ddp and str(self.device)=="cuda:0") or \
                                     (str(self.device)=="cuda") else False
        if self.on_log_device:
            pprint(self.hparams)
            pprint(self.run_info)

        

    def _init_transforms(self):
        final_size = self.hparams['final_size']
        aug_geom = [
            # v2.RandomRotation(degrees=90, fill=255, expand=True),
            # v2.Resize(size=(final_size,final_size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.25),
            v2.RandomResizedCrop(size=(final_size,final_size),
                                 scale=(0.5,1.0),
                                 ratio=(0.75,1.333), antialias=True)
        ]
        aug_color = [
            v2.RandomApply([v2.ColorJitter(
                brightness=0.5, contrast=0.8,
                saturation=0.8, hue=0.5)], p=0.8), 
            v2.RandomApply([v2.GaussianBlur(
                kernel_size=int(0.1*final_size), sigma=(0.1,2.0))],
                p=0.5), 
            v2.RandomApply([v2.Grayscale(num_output_channels=3)], p=0.2)
        ]
        base = [
            v2.ToDtype(torch.float32),
        ]
        train_transform = v2.Compose(aug_geom + aug_color + base)
        val_transform = v2.Compose(base)
        test_transform = val_transform
        self.transforms = {"train":train_transform, "val":val_transform, "test":test_transform}
        return


    def _init_dataset(self):
        dataset_name = self.run_info['dataset_name']
        # Initialize dataset
        if dataset_name=="thor":
            dataset = ObjectsDataset()
            # Split
            N = len(dataset)
            split_props = self.run_info['split_props']
            lengths = [round(split_props[0]*N),
                       round(split_props[1]*N),
                       N-round(split_props[0]*N)-round(split_props[1]*N)]
            # Random split
            train_set, val_set, test_set = torch.utils.data.random_split(
                dataset, lengths, generator=torch.Generator().manual_seed(self.SEED)
            )
            self.datasets = {"full":dataset, "train":train_set,
                             "val":val_set, "test":test_set}
            return
        # elif dataset_name=="abo_catalog":
        #     dataset = ABODataset(
        #         final_size=self.hparams['final_size'],
        #         data_root=self.run_info['data_root'],
        #         mode="catalog"
        #     )
        # elif dataset_name=="abo_spins":
        #     dataset = ABODataset(
        #         final_size=self.hparams['final_size'],
        #         data_root=self.run_info['data_root'],
        #         mode="spins"
        #     )
        elif dataset_name=="abo_catalog_per_item":
            train_set, val_set, test_set = partitionCatalogPerItem(
                split_props=self.run_info['split_props'],
                use_subset=self.run_info['use_subset']
            )
            self.datasets = {"train":train_set, "val":val_set, "test":test_set}
            return
        elif dataset_name=="abo_spins_per_object":
            train_set, val_set, test_set = partitionSpinsPerObject(
                split_props=self.run_info['split_props'],
                use_subset=self.run_info['use_subset']
            )
            self.datasets = {"train":train_set, "val":val_set, "test":test_set}
            return
        else:
            raise ValueError("Invalid/nonexistent dataset name.")

        


    def _init_loaders(self):
        self.loaders = {}
        # Unpack relevant values
        batch_size = self.hparams['batch_size']
        num_workers = self.run_info['loader_workers']
        # Create loaders
        for split in ["train","val","test"]:
            dataset = self.datasets[split]
            shuffle = True if split=="train" else False
            if self.use_ddp:
                loader = DataLoader(
                    dataset=dataset, batch_size=batch_size,
                    drop_last=False, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                    sampler=DistributedSampler(dataset), persistent_workers=True
                )
                device = int(os.environ["LOCAL_RANK"])
                print(f"On device #{device}, loader has {len(loader.sampler)}/{len(dataset)} " + 
                      f"data points in the {split} dataset.")
            else:
                loader = DataLoader(
                    dataset=dataset, batch_size=batch_size,
                    drop_last=False, shuffle=shuffle,
                    num_workers=num_workers, pin_memory=True
                )
                print(f"On device {self.device}, loader has {len(loader.sampler)}/{len(dataset)} " +
                        f"data points in the {split} dataset.")
            self.loaders[split] = loader
        return

    
    def _init_logging(self):
        trial_dir = self.run_info['trial_dir']
        # Set device for Tensorboard
        writer = None
        if self.use_ddp:
            device = int(os.environ["LOCAL_RANK"])
            if int(device)==0:
                writer = SummaryWriter(log_dir=trial_dir)
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            writer = SummaryWriter(log_dir=trial_dir)
        print(f"Tensorboard writing to {trial_dir} on device {device}.")

        self.writer = writer 
        return


    def _check_for_snapshot(self):
        snapshot_path = os.path.join(self.run_info['trial_dir'], "snapshot.pth")
        if os.path.exists(snapshot_path):
            snapshot = torch.load(snapshot_path, map_location=self.device)
            self.model.load_state_dict(snapshot["model_state"])
            self.curr_epoch = snapshot["epoch"] + 1
            self.curr_iter = snapshot["iter"] + 1
            print(f"Loading snapshot from epoch {snapshot['epoch']} / iter {snapshot['iter']}.")
            # print(f"Resuming training from epoch {self.curr_epoch} / iter {self.curr_iter}.")
        return

    # Largely copied from DDP tutorial code
    def _save_checkpoint(self):
        epoch = self.curr_epoch
        trial_dir = self.run_info['trial_dir']
        checkpoint_dict = {
            "model_state": self.model.module.state_dict() if self.use_ddp else self.model.state_dict(),
            "epoch": epoch,
            "iter": self.curr_iter,
        }
        torch.save(checkpoint_dict, str(os.path.join(trial_dir, f"snapshot.pth")))
        # print(f"Saved SNAPSHOT for epoch #{epoch}.")
        if epoch%self.run_info['save_every']==0:
            torch.save(checkpoint_dict, str(os.path.join(trial_dir, f"epoch{epoch}.pth")))
            print(f"Saved CHECKPOINT for epoch #{epoch}.")
        return

    def _init_model(self):
        # Instantiate model
        num_layers = int(self.hparams['num_layers'])
        num_bins = int(self.hparams['bins'])
        if num_layers==18:
            model = torchvision.models.resnet18()
            model.fc = nn.Linear(512,num_bins*3) # Replace final layer
        elif num_layers==34:
            model = torchvision.models.resnet34()
            model.fc = nn.Linear(512,num_bins*3) # Replace final layer
        elif num_layers==50:
            model = torchvision.models.resnet50()
            model.fc = nn.Linear(2048,num_bins*3) # Replace final layer
        else:
            print("Invalid number of layers for resnet. Try again.")
            sys.exit()
        model = model.to(self.device)
        print(f"Model created + moved to {self.device}.")
        self.model = model
        print("Model set.")

        # Check for existing snapshot
        self.curr_epoch = 0
        self.curr_iter = 0
        if self.use_ddp:
            device_id = int(os.environ["LOCAL_RANK"])
            assert self.device=="cuda:"+str(device_id)
            # # Check for existing snapshot
            self._check_for_snapshot()
            # Wrap model for DDP
            print("Wrapping model for DDP")
            self.model = DDP(self.model, device_ids=[device_id])

        print(f"Starting training at epoch #{self.curr_epoch} / {self.curr_iter} iterations")
        return

    def _init_training_objs(self):
        # Instantiate optimizer
        if self.hparams['optimizer']=="sgd":
            optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.hparams['lr']*self.hparams['lr_scaler'],
                momentum=self.hparams['momentum'],
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['optimizer']=="adam":
            optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.hparams['lr']*self.hparams['lr_scaler'],
                weight_decay=self.hparams['weight_decay']
            )
        self.optimizer = optimizer

        # Setting lr decay milestones by percentage of total iterations:
        milestones_proportions = self.hparams['lr_milestones']
        total_iters = len(self.loaders['train'])*self.hparams['total_epochs']
        milestones = [int(total_iters*prop) for prop in milestones_proportions]
        if self.on_log_device:
            print(f"Training for {total_iters} iterations at {len(self.loaders['train'])} iter per epoch, for {self.hparams['total_epochs']} epochs.")
            print(f"Sanity check: {len(self.loaders['train'])}*{self.hparams['batch_size']} =?= {len(self.loaders['train'].sampler)}")
            print(f"Milestones (in iters) = {milestones}")
        
        # Instantiate scheduler. Update last_epoch if loading from snapshot:    
        warmup_length = self.hparams['warmup_iters']
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_length,
            verbose=False)
        warmup_sched.last_epoch = self.curr_iter
        
        main_sched = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.1)
        main_sched.last_epoch = self.curr_iter
        
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched,main_sched],
            milestones=[warmup_length],
            verbose=False,
        )
        self.scheduler.last_epoch = self.curr_iter

        if self.on_log_device:
            print(f"Optimizer lr = {optimizer.param_groups[0]['lr']}")
            print(f"{warmup_sched.last_epoch=}")
            print(f"{main_sched.last_epoch=}")
            print(f"{self.scheduler.last_epoch=}")
            print(f"{self.scheduler.get_last_lr()=}")
        
        return


    def _init_normalization(self):
        dataset_name = self.run_info['dataset_name']
        loader = self.loaders['train']
        img_transform = self.transforms['train']
        bbox_transform = self.hparams['bbox_transform']
        split_props = self.run_info['split_props']

        # Get mean and std
        normalizations = [None, None]
        if self.on_log_device: # since multiple devices reading from file causes EOF error...
            if self.hparams['img_norm']=="normal": # Get into [-1,1] range
                IMG_MEAN, IMG_STD = torch.tensor(0.5), torch.tensor(0.5)
                print(f"Normal: {IMG_MEAN}, {IMG_STD}")
            elif self.hparams['img_norm']=="standard": # Use mean and std
                map_loc = f"cuda:{self.device}" if self.use_ddp else f"{self.device}"
                IMG_MEAN, IMG_STD = _get_img_normalize_values(dataset_name=dataset_name, 
                                                              loader=loader,
                                                              img_transform=img_transform,
                                                              split_props=split_props,
                                                              map_loc=map_loc)
                print(f"Standardize: {IMG_MEAN}, {IMG_STD}")
                # BBOX_MEAN, BBOX_STD = _get_bbox_normalize_values(dataset_name=dataset_name,
                #                                                  loader=loader,
                #                                                  bbox_transform=bbox_transform,
                #                                                  split_props=split_props)
            else:
                print("Invalid image normalization")
            normalizations = [IMG_MEAN, IMG_STD] #, BBOX_MEAN, BBOX_STD]
            print(f"Populated {normalizations=}")

        print(f"{normalizations=}")
        # Then broadcast to other devices    
        if self.use_ddp: 
            torch.distributed.broadcast_object_list(normalizations, src=0)
            IMG_MEAN, IMG_STD = normalizations #, BBOX_MEAN, BBOX_STD
            print("Broadcasting distributed...")
            
        # Check for invalid values, eg. NaN and Inf
        for array in [IMG_MEAN, IMG_STD]: #, BBOX_MEAN, BBOX_STD]:
            assert torch.all(torch.isfinite(array))

        self.IMG_MEAN = IMG_MEAN.to(self.device)
        self.IMG_STD = IMG_STD.to(self.device)
        # self.BBOX_MEAN = BBOX_MEAN.to(self.device)
        # self.BBOX_STD = BBOX_STD.to(self.device)
        return


    # Setup DDP, dataset, model, other training objects
    def prepare_run(self):
        # Tensorboard
        if self.on_log_device:
            self._init_logging()
            print(f"Initialized logging.")
        # Data
        self._init_dataset()
        print(f"Initialized datasets.")
        self._init_transforms()
        print(f"Initialized transforms.")
        self._init_loaders()
        print(f"Initialized dataloaders.")
        # Model + training
        self._init_model()
        print(f"Initialized model.")
        self._init_training_objs()
        print(f"Initialized training objects.")
        self._init_normalization() 
        print(f"Initialized normalization.")
        if self.use_ddp:
            torch.distributed.barrier()

    def _discretize_bboxes(self, bboxes):
        num_bins = self.hparams['bins']
        if self.hparams['bbox_transform']=="arctan":
            trans_max = torch.arctan(torch.tensor(5.0))
            trans_min = torch.arctan(torch.tensor(0.0))
            trans_labels = torch.arctan(bboxes)
        elif self.hparams['bbox_transform']=="none":
            trans_max = torch.tensor(5.0)
            trans_min = torch.tensor(0.0)
            trans_labels = bboxes
        trans_bins = torch.linspace(start=trans_min,
                                    end=trans_max,steps=num_bins+1,
                                    device=self.device)
        classes = torch.searchsorted(sorted_sequence=trans_bins, 
                                     input=trans_labels)
        classes = torch.clamp(classes, min=1, max=num_bins) - 1
        assert classes.shape==bboxes.shape
        return classes

    def _verify_preprocessed_imgs(self, orig_imgs, trans_imgs, split, labels, predictions):
        # Convert logits to class predictions of shape (batch,3).
        # Take small subset for visualization.
        k = 20
        predictions = torch.argmax(predictions, dim=1)[0:k]
        labels = labels[0:k]
        trans_imgs = trans_imgs[0:k]
        # Create matplot figure, populate with image and title as prediction+label
        ncols=4
        nrows=int(k/4)
        fig,axes = plt.subplots(ncols=ncols, nrows=nrows, num=1, clear=True)
        fig.set_size_inches(2.5*ncols,2.5*nrows)
        axes = axes.flatten()
        for i,(pred,label,img,ax) in enumerate(zip(predictions,labels,trans_imgs,axes)):
            # if split=="train":
            img = img.type(torch.uint8)
            img = torchvision.transforms.functional.to_pil_image(img)
            ax.imshow(img)
            prediction_str = f"{str(pred.cpu().tolist())}->{str(label.cpu().tolist())}"
            ax.set_title(label=prediction_str)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        fig.suptitle(f"Accuracy peek on {k} imgs: (pred)->(label)")
        fig.tight_layout()
        self.writer.add_figure(tag=f"accuracy_peek ({split})",figure=fig,global_step=self.curr_iter)
        fig.clear()
        plt.close(fig) 
        return
    
    def _preprocess_imgs(self, imgs, split):
        transform = self.transforms[split]
        trans_imgs = torch.stack([transform(img) for img in imgs]) # uint8[0,255] --> float[0.,255.]
        assert torch.all(torch.isfinite(trans_imgs))
        # float[0.,255.] --> float[0.,1.] --> float[-1.,1.]
        norm_imgs = (trans_imgs/255. - self.IMG_MEAN) / (self.IMG_STD + 1e-6)
        assert torch.all(torch.isfinite(norm_imgs))
        return trans_imgs, norm_imgs

        
    def _sync(self, metric):
        gathered_metrics = [torch.zeros_like(metric) for _ in range(self.gpus_per_worker)]
        torch.distributed.all_gather(tensor=metric, tensor_list=gathered_metrics)
        final_metric = sum(gathered_metrics)
        return final_metric
    
    def _val_epoch(self):
        total_loss = torch.tensor(0.0, device=self.device)
        total_num_correct = torch.tensor([0,0,0], device=self.device)

        loader = self.loaders["val"]
        num_bins = self.hparams['bins']
        
        # Loop through dataloader
        for i,(imgs,bboxes) in enumerate(tqdm(loader)):
            orig_imgs = imgs.to(self.device, non_blocking=True)
            bboxes = bboxes.to(self.device, non_blocking=True)
            
            # Transform and normalize images, normalize labels
            trans_imgs, norm_imgs = self._preprocess_imgs(orig_imgs, split="val")
            class_labels = self._discretize_bboxes(bboxes)
            
            # Forward pass
            output_logits = self.model(norm_imgs)
            output_logits = torch.reshape(output_logits, (-1,num_bins,3))
            output = torch.argmax(output_logits, dim=1) # Class prediction
            assert torch.all(torch.isfinite(output_logits))
            assert int(output.max())<=int(num_bins-1)
            assert int(output.min())>=0
            loss = F.cross_entropy(input=output_logits, target=class_labels)

            # Accuracy peek + visualization
            if self.on_log_device and i==0:
                self._verify_preprocessed_imgs(orig_imgs[::12], trans_imgs[::12], "val", class_labels[::12], output_logits[::12])
            
            # Accumulate loss and # correct classifications over entire epoch
            total_loss += loss.detach().clone()*orig_imgs.shape[0]
            total_num_correct += torch.sum(
                torch.eq(output, class_labels).long(),
                dim=0
            ).detach().clone()

        if self.use_ddp:
            total_loss = self._sync(total_loss)
            total_num_correct = self._sync(total_num_correct)
        return total_loss, total_num_correct


    def _train_epoch(self):
        total_loss = torch.tensor(0.0, device=self.device)
        total_num_correct = torch.tensor([0,0,0], device=self.device)

        loader = self.loaders["train"]
        num_bins = self.hparams['bins']
        
        # Loop through dataloader
        for i,(imgs,bboxes) in enumerate(tqdm(loader)):
            orig_imgs = imgs.to(self.device, non_blocking=True)
            bboxes = bboxes.to(self.device, non_blocking=True)
            
            # Transform and normalize images, normalize labels
            trans_imgs, norm_imgs = self._preprocess_imgs(orig_imgs, split="train")
            class_labels = self._discretize_bboxes(bboxes)
            
            # Forward pass
            output_logits = self.model(norm_imgs)
            output_logits = torch.reshape(output_logits, (-1,num_bins,3))
            output = torch.argmax(output_logits, dim=1) # Class prediction

            assert torch.all(torch.isfinite(output_logits))
            assert int(output.max())<=int(num_bins-1)
            assert int(output.min())>=0
            loss = F.cross_entropy(input=output_logits, target=class_labels)

            # Visualize image inputs
            with torch.no_grad():
                if self.on_log_device and i==0:
                    self._verify_preprocessed_imgs(orig_imgs, trans_imgs, "train", class_labels, output_logits)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                           max_norm=1.0, norm_type=2.0,
                                           error_if_nonfinite=True)
            self.optimizer.step()

            # Iteration-level lr scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Track iteration loss
            if self.on_log_device:
                self.writer.add_scalar(
                    f"iter_loss/train",
                    loss, global_step=self.curr_iter)
                self.writer.add_scalar(
                    f"lr",
                    self.scheduler.get_last_lr()[0],
                    global_step=self.curr_iter)
            # Accumulate loss and # correct classifications over entire epoch
            total_loss += loss.detach().clone()*orig_imgs.shape[0]
            total_num_correct += torch.sum(
                torch.eq(output, class_labels).long(),
                dim=0
            ).detach().clone()    
            self.curr_iter += 1

        if self.use_ddp:
            total_loss = self._sync(total_loss)
            total_num_correct = self._sync(total_num_correct)
        return total_loss, total_num_correct


    # Average sums over number of data points, eg. average loss per image
    def _process_metrics(self, split, epoch_loss_sum, total_correct):
        denom = len(self.datasets[split])
        print(f"denom for {split} split = {denom}")
        loss = (epoch_loss_sum / denom).item()
        acc_short, acc_med, acc_long = (total_correct / denom).cpu().numpy()
        acc_total = (torch.sum(total_correct)/(3*denom)).item()
        
        metrics = {
            f'avg_loss/{split}':loss,
            f'acc_short/{split}':acc_short,
            f'acc_med/{split}':acc_med,
            f'acc_long/{split}':acc_long,
            f'acc_total/{split}':acc_total
        }
        return metrics

    # Write metrics to Tensorboard + hparams
    def _save_metrics(self, metrics):
        for k,v in metrics.items():
            self.writer.add_scalar(k, v, global_step=self.curr_epoch)
        self.writer.add_hparams(
            hparam_dict=self.hparams,
            metric_dict={str("z_"+k):v for k,v in metrics.items()},
            run_name="."
        )
        return
    
    # Training loop
    def execute_run(self):
        start_epoch = self.curr_epoch
        end_epoch = self.hparams['total_epochs']
        for epoch in range(start_epoch, end_epoch):
            assert epoch==self.curr_epoch
            start_time = time()

            # One pass over dataset
            if self.use_ddp:
                self.loaders['train'].sampler.set_epoch(self.curr_epoch)
            self.model.train()
            train_epoch_loss_sum, train_total_num_correct = self._train_epoch()
            self.model.eval()
            with torch.no_grad():
                val_epoch_loss_sum, val_total_num_correct = self._val_epoch()
            
            if self.on_log_device:
                # Get average loss over entire epoch / entire dataset,
                train_metrics = self._process_metrics(
                    "train", train_epoch_loss_sum, train_total_num_correct
                )
                val_metrics = self._process_metrics(
                    "val", val_epoch_loss_sum, val_total_num_correct
                )
                # Merge per-split metric dictionaries into one
                all_metrics = train_metrics | val_metrics | {"time_per_epoch": time() - start_time}
                # log metrics and hyperparams, 
                self._save_metrics(all_metrics)
                self.writer.flush()
                # and save checkpoint
                self._save_checkpoint()
            self.curr_epoch += 1

            if self.use_ddp:
                torch.distributed.barrier()
                print("Distributed barrier.")
            # END OF EPOCH


        # Training complete, cleanup
        if self.use_ddp:
            print("Destroying process group.")
            destroy_process_group()
        if self.on_log_device:
            print("Closing Tensorboard writer.")
            self.writer.close()
        return




# TODO: normalize_imgs, discretize_bboxes, accuracy_peek, save_checkpoint
# LATER: write_hparams

# accuracy_peek --> global percentage accuracy for train&val + one val minibatch w/ img visualization and observing prediction wiggle


if __name__=="__main__":
    # Prepare save folder
    exps_dir = os.environ['EXPS_DIR']
    exp_name = os.environ['EXP_NAME']
    trial_num = os.environ['TRIAL_NUM']
    trial_dir = os.path.join(exps_dir, exp_name, trial_num)
    os.makedirs(name=trial_dir, exist_ok=True)
    trial_num = int(trial_num)
    # Load hparam and other run info
    with open(os.path.join(exps_dir, exp_name, "hparams.json")) as f:
        all_hparams = json.load(f)
        hparam_dict = all_hparams[trial_num]
    with open(os.path.join(exps_dir, exp_name, "run_info.json")) as f:
        run_info_dict = json.load(f)
        assert run_info_dict['exp_name']==exp_name
    # Add trial_num information
    run_info_dict['trial_num'] = trial_num
    run_info_dict['trial_dir'] = trial_dir
    run_info_dict['exps_dir'] = exps_dir
    hparam_dict['lr_milestones'] = torch.tensor(hparam_dict['lr_milestones'])
    
    # Do the training
    print(f"Training trial number {trial_num}...")
    run = TrainingRun(hparams=hparam_dict, run_info=run_info_dict)
    run.prepare_run()
    run.execute_run()
    print("Training run complete.")
