# Utilities
import os
# import argparse
import sys
import pdb
import numpy as np
from pprint import pprint
# import matplotlib.pyplot as plt

# Import training libraries: torch, tqdm, tensorboard
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
torchvision.disable_beta_transforms_warning()
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

from datasets import ABODataset, ObjectsDataset
from utils_OOP import _get_img_normalize_values, _get_bbox_normalize_values


class TrainingRun():
    def __init__(self, hparams, run_info):
        super().__init__()
        self.SEED = SEED
        self.hparams = hparams
        self.run_info = run_info
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Scale lr as needed wrt batch size and number of GPUs
        bs = hparams['batch_size']
        orig_lr = hparams['lr']
        new_lr = orig_lr
        # 1. Per-device lr scaling w.r.t. default batch size of 256
        if bs > 256:
            lr_multiplier = int(bs / 256)
            new_lr = new_lr * lr_multiplier
        # 2. Further scaling if using multiple GPUs
        if run_info['gpus_per_worker'] > 1:
            new_lr = new_lr * run_info['gpus_per_worker']
        self.hparams['lr'] = new_lr
        print(f"Effective batch size: {bs} * {run_info['gpus_per_worker']} = {bs*run_info['gpus_per_worker']}")
        print(f"Learning rate scaled by {int(new_lr/orig_lr)}*{orig_lr} to {self.hparams['lr']}.")

    
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
                brightness=0.8, contrast=0.8,
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
        elif dataset_name=="abo_catalog":
            dataset = ABODataset(final_size=self.hparams['final_size'], mode="catalog")
        elif dataset_name=="abo_spins":
            dataset = ABODataset(final_size=self.hparams['final_size'], mode="spins")
        else:
            raise ValueError("Invalid/nonexistent dataset name.")

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


    def _init_loaders(self):
        self.loaders = {}
        # Unpack relevant values
        batch_size = self.hparams['batch_size']
        num_workers = self.run_info['num_workers']
        # Create loaders
        for split in ["train","val","test"]:
            dataset = self.datasets[split]
            shuffle = True if split=="train" else False
            loader = DataLoader(
                dataset=dataset, batch_size=batch_size,
                drop_last=False, shuffle=shuffle,
                num_workers=num_workers, pin_memory=True
            )
            print(f"On device {self.device}, loader has {len(loader.sampler)}/{len(dataset)} " +
                    f"data points in the {split} dataset.")
            self.loaders[split] = loader
        return

    
    # def _init_logging(self):
    #     # print(f"{trial_num=}")
    #     existing_trials = os.listdir("/home/jovyan/trials")
    #     # existing_trials = [int(subdir) for subdir in existing_trials]
    #     # trial_dir = f"./trials/{self.trial_num}"
    #     trial_dir = f"/home/jovyan/trials/{len(existing_trials) + 1}"
            
    #     # Set device for Tensorboard
    #     writer = None
    #     if self.use_ddp:
    #         _ddp_setup()
    #         device = int(os.environ["LOCAL_RANK"])
    #         if int(device)==0:
    #             writer = SummaryWriter(log_dir=trial_dir)
    #         torch.distributed.barrier()
    #     else:
    #         device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #         writer = SummaryWriter(log_dir=trial_dir)
    #     print(f"Tensorboard writing to {trial_dir} on device {device}.")

    #     self.trial_dir = trial_dir
    #     self.writer = writer 
    #     self.device = device
    #     self.on_log_device = True if str(self.device)=="0" or self.device=="cuda" else False
        
    #     # Setting up filesystem to store results
    #     if self.use_ddp:
    #         if not os.path.isdir(trial_dir) and self.on_log_device:
    #             os.makedirs(trial_dir)
    #     else:
    #         if not os.path.isdir(trial_dir):
    #             os.makedirs(trial_dir)
    #     return


    # def _check_for_snapshot(self):
    #     snapshot_path = os.path.join(self.trial_dir, "snapshot.pth")
    #     if os.path.exists(snapshot_path):
    #         snapshot = torch.load(snapshot_path, map_location=f"cuda:{self.device}")
    #         self.model.load_state_dict(snapshot["model_state"])
    #         self.curr_epoch = snapshot["epoch"] + 1
    #         self.curr_iter = snapshot["iter"] + 1
    #         print(f"Loading snapshot from epoch {snapshot['epoch']} / iter {snapshot['iter']}.")
    #         print(f"Resuming training from epoch {self.curr_epoch} / iter {self.curr_iter}.")
    #     return

    # # Largely copied from DDP tutorial code
    # def _save_checkpoint(self):
    #     epoch = self.curr_epoch
    #     checkpoint_dict = {
    #         "model_state": self.model.module.state_dict() if self.use_ddp else self.model.state_dict(),
    #         "epoch": epoch,
    #         "iter": self.curr_iter,
    #     }
    #     torch.save(checkpoint_dict, str(os.path.join(self.trial_dir, f"snapshot.pth")))
    #     print(f"Saved SNAPSHOT for epoch #{epoch}.")
    #     if epoch%self.run_info['save_every']==0:
    #         torch.save(checkpoint_dict, str(os.path.join(self.trial_dir, f"epoch{epoch}.pth")))
    #         print(f"Saved CHECKPOINT for epoch #{epoch}.")
    #     return

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

        # Check for existing snapshot
        self.curr_epoch = 0
        self.curr_iter = 0
        # if self.use_ddp:
        #     assert self.device==int(os.environ["LOCAL_RANK"])
        #     # # Check for existing snapshot
        #     # self._check_for_snapshot()
        #     # Wrap model for DDP
        #     self.model = DDP(self.model, device_ids=[self.device])

        print(f"Starting training at epoch #{self.curr_epoch} / {self.curr_iter} iterations")
        return

    def _init_training_objs(self):
        # Instantiate optimizer
        if self.hparams['optimizer']=="sgd":
            optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.hparams['lr'],
                momentum=self.hparams['momentum'],
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['optimizer']=="adam":
            optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.hparams['lr'],
                weight_decay=self.hparams['weight_decay']
            )
        self.optimizer = optimizer

        # Setting lr decay milestones by percentage of total iterations:
        milestones_proportions = self.hparams['lr_milestones']
        total_iters = len(self.loaders['train'])*self.hparams['total_epochs']
        print(f"Training for {total_iters} iterations at {len(self.loaders['train'])} iter per epoch, for {self.hparams['total_epochs']} epochs.")
        print(f"Sanity check: {len(self.loaders['train'])}*{self.hparams['batch_size']} =?= {len(self.loaders['train'].sampler)}")
        milestones = [int(total_iters*prop) for prop in milestones_proportions]
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
        # if self.on_log_device: # since multiple devices reading from file causes EOF error...
        if self.hparams['img_norm']=="normal": # Get into [-1,1] range
            IMG_MEAN, IMG_STD = torch.tensor(0.5), torch.tensor(0.5)
        elif self.hparams['img_norm']=="standard": # Use mean and std
            # map_loc = f"cuda:{self.device}" if self.use_ddp else f"{self.device}"
            map_loc = self.device
            IMG_MEAN, IMG_STD = _get_img_normalize_values(dataset_name=dataset_name, 
                                                          loader=loader,
                                                          img_transform=img_transform,
                                                          split_props=split_props,
                                                          map_loc=map_loc)
            # BBOX_MEAN, BBOX_STD = _get_bbox_normalize_values(dataset_name=dataset_name,
            #                                                  loader=loader,
            #                                                  bbox_transform=bbox_transform,
            #                                                  split_props=split_props)
        normalizations = [IMG_MEAN, IMG_STD] #, BBOX_MEAN, BBOX_STD]
        # else:
        #     normalizations = [None, None] #, None, None]
        
        # Then broadcast to other devices    
        # if self.use_ddp: 
        #     torch.distributed.broadcast_object_list(normalizations, src=0)
        #     IMG_MEAN, IMG_STD = normalizations #, BBOX_MEAN, BBOX_STD
        
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
        # self._init_logging()
        # Data
        self._init_dataset()
        self._init_transforms()
        self._init_loaders()
        # Model + training
        self._init_model()
        self._init_training_objs()
        self._init_normalization() 

        # if self.use_ddp:
        #     torch.distributed.barrier()

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

    def _verify_preprocessed_imgs(self, imgs, trans_imgs):
        pdb.set_trace()
        for i,(pre_img,trans_img) in enumerate(zip(imgs, trans_imgs)):
            pre_img = torchvision.transforms.functional.to_pil_image(pre_img)
            pre_img.save(f"/home/jovyan/pre{i}.png")
            trans_img = torchvision.transforms.functional.to_pil_image(trans_img.type(torch.uint8))
            trans_img.save(f"/home/jovyan/trans{i}.png")
            if i > 10:
                break
        return
    
    def _preprocess_imgs(self, imgs, split):
        transform = self.transforms[split]
        trans_imgs = torch.stack([transform(img) for img in imgs]) # uint8[0,255] --> float[0.,255.]
        # self._verify_preprocessed_imgs(imgs,trans_imgs)
        assert torch.all(torch.isfinite(trans_imgs))
        # float[0.,255.] --> float[0.,1.] --> float[-1.,1.]
        norm_imgs = (trans_imgs/255. - self.IMG_MEAN) / (self.IMG_STD + 1e-6)
        assert torch.all(torch.isfinite(norm_imgs))
        return trans_imgs, norm_imgs

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

            # Track loss
            total_loss += loss.detach().clone()*orig_imgs.shape[0]
            total_num_correct += torch.sum(
                torch.eq(output, class_labels).long(),
                dim=0
            ).detach().clone()

        # if self.use_ddp:
        #     torch.distributed.barrier()
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

            # # Track loss
            # if self.on_log_device:
            #     self.writer.add_scalar(f"iter_loss/train", loss, global_step=self.curr_iter)
            #     self.writer.add_scalar(f"lr", self.scheduler.get_last_lr()[0], global_step=self.curr_iter)
            total_loss += loss.detach().clone()*orig_imgs.shape[0]
            total_num_correct += torch.sum(
                torch.eq(output, class_labels).long(),
                dim=0
            ).detach().clone()

            # if self.use_ddp:
            #     torch.distributed.barrier()      
            self.curr_iter += 1

        return total_loss, total_num_correct



    def _log_epoch_metrics(self, metric_name, train_partial_sum, val_partial_sum):
        if self.use_ddp:
            # Syncing loss values between devices:
            print(f"On device {self.device}, partial {metric_name}'s numerator = {train_partial_sum}")
            print(f"On device {self.device}, partial {metric_name}'s numerator = {val_partial_sum}")

            train_epoch_metrics = [torch.zeros_like(train_partial_sum)
                                  for _ in range(torch.cuda.device_count())]
            val_epoch_metrics = [torch.zeros_like(val_partial_sum)
                                for _ in range(torch.cuda.device_count())]

            torch.distributed.all_gather(
                tensor=train_partial_sum, tensor_list=train_epoch_metrics)
            torch.distributed.all_gather(
                tensor=val_partial_sum, tensor_list=val_epoch_metrics)
            torch.distributed.barrier()

            train_avg_metric = sum(train_epoch_metrics) / len(self.datasets['train'])
            val_avg_metric = sum(val_epoch_metrics) / len(self.datasets['val'])
            
        else:
            train_avg_metric = train_partial_sum / len(self.datasets['train'])
            val_avg_metric = val_partial_sum / len(self.datasets['val'])
            # self.writer.add_scalar(f"{metric_name}/train", train_avg_metric, global_step=self.curr_epoch)
            # self.writer.add_scalar(f"{metric_name}/val", val_avg_metric, global_step=self.curr_epoch)
        
        if self.on_log_device and not self.use_ray_tune:
            if metric_name=="epoch_avg_accuracy":
                self.writer.add_scalars(f"{metric_name}/train",
                                        {'short':train_avg_metric[0],
                                         'medium':train_avg_metric[1],
                                         'long':train_avg_metric[2]},
                                        global_step=self.curr_epoch)
                self.writer.add_scalars(f"{metric_name}/val",
                                        {'short':val_avg_metric[0],
                                         'medium':val_avg_metric[1],
                                         'long':val_avg_metric[2]},
                                        global_step=self.curr_epoch)
                self.writer.add_hparams(hparam_dict=self.hparams, 
                                        metric_dict={
                                            f"{metric_name}-train":torch.mean(train_avg_metric),
                                            f"{metric_name}-val":torch.mean(val_avg_metric),
                                        },
                                        run_name=".")
            elif metric_name=="epoch_avg_loss":
                self.writer.add_scalar(f"{metric_name}/train",
                                       train_avg_metric,
                                       global_step=self.curr_epoch)
                self.writer.add_scalar(f"{metric_name}/val",
                                       val_avg_metric,
                                       global_step=self.curr_epoch)
                self.writer.add_hparams(hparam_dict=self.hparams, 
                                        metric_dict={
                                            f"{metric_name}-train":train_avg_metric,
                                            f"{metric_name}-val":val_avg_metric,
                                        },
                                        run_name=".")
        return train_avg_metric, val_avg_metric

    # Training loop
    def execute_run(self):
        start_epoch = self.curr_epoch
        end_epoch = self.hparams['total_epochs']
        for epoch in range(start_epoch, end_epoch):
            assert epoch==self.curr_epoch
            start_time = time()

            # One pass over dataset
            # if self.use_ddp:
            #     self.loaders['train'].sampler.set_epoch(self.curr_epoch)
            self.model.train()
            train_epoch_loss_sum, train_total_correct = self._train_epoch()
            self.model.eval()
            with torch.no_grad():
                val_epoch_loss_sum, val_total_correct = self._val_epoch()

            train_denom = len(self.datasets['train'])
            train_loss = (train_epoch_loss_sum / train_denom).item()
            train_short, train_med, train_long = (train_total_correct / train_denom).cpu().numpy()
            train_accuracy = (torch.sum(train_total_correct)/(3*train_denom)).item()
            # train_accuracy = (torch.sum(train_total_num_correct).float()/torch.numel(train_total_num_correct)).item()
            
            val_denom = len(self.datasets['val'])
            val_loss = (val_epoch_loss_sum / val_denom).item()
            val_short, val_med, val_long = (val_total_correct / val_denom).cpu().numpy()
            val_accuracy = (torch.sum(val_total_correct)/(3*val_denom)).item()
            
            # Get average loss over entire epoch / entire dataset,
            # log hyperparams, and save checkpoint
            # if self._on_log_device:
            # with torch.no_grad():
            #     train_avg_loss, val_avg_loss = self._log_epoch_metrics("epoch_avg_loss",
            #                                                            train_epoch_loss_sum,
            #                                                            val_epoch_loss_sum)
            #     train_avg_accuracy, val_avg_accuracy = self._log_epoch_metrics("epoch_avg_accuracy",
            #                                                                    train_total_num_correct,
            #                                                                    val_total_num_correct)
            #     if self.on_log_device:
            #         self.writer.add_scalar("time_per_epoch", time()-start_time, global_step=self.curr_epoch)
            # # if self.on_log_device:
            #         self.writer.flush()
            #         self._save_checkpoint()

            
            # Ray tune: report metrics + save checkpoint
            checkpoint_data = {
                "epoch": self.curr_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)
    
                checkpoint = Checkpoint.from_directory(checkpoint_dir)

                train.report(
                    {"train_loss":train_loss,
                     "train_accuracy": train_accuracy,
                     "train_short":train_short,
                     "train_med":train_med,
                     "train_long":train_long,
                     "val_loss": val_loss,
                     "val_accuracy": val_accuracy,
                     "val_short":val_short,
                     "val_med":val_med,
                     "val_long":val_long,
                     "current_lr":self.scheduler.get_last_lr()[0]},
                    checkpoint=checkpoint,
                )
            assert epoch==self.curr_epoch
            self.curr_epoch += 1

        return




# TODO: normalize_imgs, discretize_bboxes, accuracy_peek, save_checkpoint
# LATER: write_hparams

# accuracy_peek --> global percentage accuracy for train&val + one val minibatch w/ img visualization and observing prediction wiggle

def run_one_trial(config, run_info_dict):
    run = TrainingRun(hparams=config, run_info=run_info_dict)
    run.prepare_run()
    run.execute_run()
    return

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument("--trial_num",
                        help="Trial number within one experiment.",
                        type=int, default=1)
    parser.add_argument("--lr",
                        help="Starting learning rate.",
                        type=float, nargs='+', default=[0.1])
    parser.add_argument("--batch_size",
                        help="Batch size per GPU.",
                        type=int, nargs='+', default=[256])
    parser.add_argument("--total_epochs",
                        help="Total epochs in one training run.",
                        type=int, nargs='+', default=[10])
    parser.add_argument("--warmup_iters",
                        help="Number of iterations in linear warmup.",
                        type=int, nargs='+', default=[500])
    parser.add_argument("--lr_milestones",
                        help="Proportions of training run (out of 1.0) at which to decay learning rate.",
                        nargs='+', default=['[0.5,0.75]'])
    parser.add_argument("--momentum",
                        help="SGD momentum.",
                        type=float, nargs='+', default=[0.9])
    parser.add_argument("--weight_decay",
                        help="Weight decay.",
                        type=float, nargs='+', default=[0.0])
    parser.add_argument("--num_layers",
                        help="Number of layers in Resnet.",
                        type=int, nargs='+', default=[18])
    parser.add_argument("--optimizer",
                        help="Choice of optimizer, eg. SGD vs Adam.",
                        type=str, nargs='+', default=["sgd"])
    
    parser.add_argument("--bins",
                        help="Number bins for classification.",
                        type=int, nargs='+', default=[50])
    parser.add_argument("--bbox_transform",
                        help="Transformation applied to bbox values before bucketing.",
                        type=str, nargs='+', default=["none"])
    parser.add_argument("--final_size",
                        help="Resolution of image input to model.",
                        type=int, nargs='+', default=[256])
    parser.add_argument("--img_norm",
                        help="Preprocess with standardization or normalization.",
                        type=str, nargs='+', default=["normal"])
    
    # Run info
    parser.add_argument("--dataset_name",
                        help="Dataset name: ABO360 ('abo_spins', 'abo_catalog').",
                        type=str, default="abo_spins")
    parser.add_argument("--split_props",
                        help="Train/val/test proportions.",
                        type=float, nargs='+', default=[0.7,0.1,0.2])
    parser.add_argument("--num_loader_workers",
                        help="Number of workers in one Dataloader.",
                        type=int, default=8)
    parser.add_argument("--num_ray_workers",
                        help="Number of parallel trials Ray tune should run.",
                        type=int, default=torch.cuda.device_count())
    parser.add_argument("--cpus_per_worker",
                        help="Number of CPUs per Ray tune trial.",
                        type=int, default=1)
    parser.add_argument("--gpus_per_worker",
                        help="Number of GPUs per Ray tune trial.",
                        type=int, default=1)
    parser.add_argument("--use_ddp",
                        help="Flat to toggle use of DDP for distributed training.",
                        action="store_true", default=False)
    # parser.add_argument("--use_ray_tune",
    #                     help="Perform parallel hyperparam search with Ray tune.",
    #                     action="store_true", default=False)
    parser.add_argument("--total_trials",
                        help="Number of trials in hparam search.",
                        type=int, default=0)
    args = parser.parse_args()
    run_info_dict = {
        'dataset_name':args.dataset_name,
        'split_props': args.split_props,
        'save_every':1,
        'num_workers':args.num_loader_workers,
        'use_ddp':args.use_ddp,
        'num_gpus':torch.cuda.device_count(),
        'num_ray_workers':args.num_ray_workers,
        'cpus_per_worker':args.cpus_per_worker,
        'gpus_per_worker':args.gpus_per_worker,
    }
    import ast
    # if args.use_ray_tune:
    import tempfile
    from pathlib import Path
    from functools import partial
    import ray
    from ray import train, tune
    from ray.train import Checkpoint
    from ray.tune.schedulers import ASHAScheduler
    import ray.cloudpickle as pickle
    from ray.train.torch import TorchTrainer, get_device, get_devices

        
    hparam_config = {
        'lr':tune.choice(args.lr),
        'batch_size':tune.choice(args.batch_size),
        'total_epochs':tune.choice(args.total_epochs),
        'warmup_iters':tune.choice(args.warmup_iters),
        'lr_milestones':tune.choice(
            torch.tensor([ast.literal_eval(milestone) for milestone in args.lr_milestones])),
        'momentum':tune.choice(args.momentum),
        'weight_decay':tune.choice(args.weight_decay),
        'num_layers':tune.choice(args.num_layers),
        'bins':tune.choice(args.bins),
        'bbox_transform':tune.choice(args.bbox_transform),
        'final_size':tune.choice(args.final_size),
        'img_norm':tune.choice(args.img_norm),
        'optimizer':tune.choice(args.optimizer),
    }
    if args.total_trials==0:
        num_combs = np.prod([len(v) for k,v in hparam_config.items()])
        # print(f"Current hparam_config has {num_combs} unique hparam combinations.")
        args.total_trials = num_combs
    print(f"Experiment has {args.total_trials} total trials.")
    
    # Introducing TorchTrainer do automatically allow DDP capabilities
    from ray.train import RunConfig, ScalingConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer
    scaling_config = ScalingConfig(
        # num_workers=1, use_gpu=True, resources_per_worker={"CPU": 2, "GPU": 1}
        num_workers=args.num_ray_workers,
        use_gpu=True,
        resources_per_worker={
            "CPU": args.cpus_per_worker,
            "GPU": args.gpus_per_worker
        }
    )
    run_config = RunConfig(
        # checkpoint_config=CheckpointConfig(
        #     num_to_keep=2,
        #     checkpoint_score_attribute="ptl/val_accuracy",
        #     checkpoint_score_order="max",
        # ),
    )
    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        partial(run_one_trial, run_info_dict=run_info_dict),
        scaling_config=scaling_config,
        run_config=run_config,
    )
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=max(args.total_epochs),
        grace_period=1,
        reduction_factor=2,
    )

    # hparam optimization object
    tuner = tune.Tuner(
        # hparam objective function
        trainable=ray_trainer,
        # hparam optimization search space:
        param_space={"train_loop_config": hparam_config},
        # hyperparameter optimization goal:
        tune_config=tune.TuneConfig(
            # metric="val_loss",
            # mode="min",
            num_samples=args.total_trials,
            scheduler=scheduler,
        ),
    )
    results = tuner.fit()

    # results = tune.run(
    #     partial(run_one_trial, run_info_dict=run_info_dict), 
    #     resources_per_trial={"cpu": 4, "gpu": 1},
    #     config=hparam_config,
    #     num_samples=args.total_trials,
    #     scheduler=scheduler,
    #     # checkpoint_at_end=True
    # )
    best_result = results.get_best_result(metric="val_loss",
                                       mode="min",
                                       scope="all")
    df = results.get_dataframe(filter_metric="val_loss", filter_mode="min")
    wanted_metrics = ["val_loss", "val_accuracy", "train_loss", "train_accuracy"]
    display_cols = []
    for col in df.columns:
        # print(col)
        do_display = False
        for wanted in wanted_metrics:
            if wanted in col:
                do_display = True
        if do_display:
            display_cols.append(col)
    # print(display_cols)
    print(df[display_cols])
    # print(f"{best_trial.trainable_name=}")
    # print(f"{best_trial.trial_id=}")
        
    # # Manually loop through all hparam combinations
    # else:
    #     # Cartesian product of all possible hyperparam values, ie. grid search
    #     import itertools
    #     hyperparam_grid = [
    #         args.lr,             # 0
    #         args.batch_size,     # 1  
    #         args.total_epochs,   # 2
    #         args.warmup_iters,   # 3
    #         args.lr_milestones,  # 4
    #         args.weight_decay,   # 5
    #         args.bins,           # 6
    #         args.bbox_transform, # 7
    #         args.final_size,     # 8
    #         args.img_norm,       # 9
    #         args.optimizer,       # 10
    #         args.momentum,        # 11
    #         args.num_layers      # 12
    #     ]
    #     for hparam_instance in itertools.product(*hyperparam_grid):
    #         print(hparam_instance)
    #     print()
        
    #     trial_num = args.trial_num
    #     for hparam_instance in itertools.product(*hyperparam_grid):
    #         hparam_dict={
    #             'lr':hparam_instance[0],
    #             'batch_size':hparam_instance[1],
    #             'total_epochs':hparam_instance[2],
    #             'warmup_iters':hparam_instance[3],
    #             'lr_milestones':torch.tensor(ast.literal_eval(hparam_instance[4])),
    #             'momentum':hparam_instance[11],
    #             'weight_decay':hparam_instance[5],
    #             'num_layers':hparam_instance[12],
    #             'bins':hparam_instance[6],
    #             'bbox_transform':hparam_instance[7],
    #             'final_size':hparam_instance[8],
    #             'img_norm':hparam_instance[9],
    #             'optimizer':hparam_instance[10]
    #         }
    #         if hparam_dict['optimizer']=="adam":
    #             hparam_dict['momentum'] = 0.0
                
    #         run_info_dict = {
    #             'dataset_name':args.dataset_name,
    #             'split_props': args.split_props,
    #             # 'trial_num':trial_num,
    #             'save_every':1,
    #             'num_workers':args.num_workers,
    #             'use_ddp':args.use_ddp,
    #             'num_gpus':torch.cuda.device_count()
    #         }
    #         # assert hparam_dict['total_epochs']==args.total_epochs
    #         assert run_info_dict['use_ddp']==args.use_ddp
    #         # assert run_info_dict['trial_num']==trial_num
    #         assert sum(run_info_dict['split_props'])==1.0

    #         # run_one_trial(hparam_dict, run_info_dict, trial_num)
    #         run = TrainingRun(hparams=hparam_dict, run_info=run_info_dict)
    #         run.prepare_run()
    #         run.execute_run()
    #         trial_num += 1

