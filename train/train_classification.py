import torch
import torch.nn as nn
import sys
import time
import cv2
import os
import datetime
sys.path.append("..")
from torch.utils.data import DataLoader
from Datasets.DataLoader import Img_DataLoader
from utils.utils import configure_optimizers
class trainer_classification(nn.Module):
    def __init__(self, train_image_files, validation_image_files, train_labels, validation_labels, gamma = 0.1,
                               init_lr = 0.001, weight_decay = 0.0005, batch_size = 32, epochs = 30, lr_decay_every_x_epochs = 10,
                 print_steps = 50, df = None, img_transform = False, model =False,
                save_checkpoints_dir = None):
        super(trainer_classification, self).__init__()
        assert model != False, 'Please put a model!'
        assert img_transform != False, 'Please put a augumentation pipeline!'
        self.df = df
        names = list(set(self.df['Cell_Types'].tolist()))
        
        self.train_image_files = train_image_files
        self.validation_image_files = validation_image_files
        self.train_labels = train_labels
        self.validation_labels = validation_labels
        self.batch_size = batch_size
        self.epoch = epochs
        self.global_step = 0
        self.current_step = 0
        self.init_lr = init_lr
        self.lr_decay_every_x_epochs = lr_decay_every_x_epochs
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.print_steps = print_steps
        self.img_transform = img_transform
        self.model = model
        self.save_checkpoints_dir = save_checkpoints_dir
        self.date = datetime.datetime.now()


    def _dataloader(self, datalist, labels, split='train',img_transform = False):
        dataset = Img_DataLoader(img_list=datalist, labels=labels, split=split, transform = img_transform, df = self.df)
        shuffle = True if split == 'train' else False
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=2, shuffle=shuffle)
        return dataloader

    def train_one_epoch(self, epoch, train_loader, model, optimizer, lr_scheduler):
        t0 = 0.0
        model.train()
        for inputs in train_loader:
            self.global_step += 1
            self.current_step +=1

            t1 = time.time()

            batch_images = inputs["image"].cuda()
            ground_truths = inputs["label"]
            ground_truths = ground_truths.reshape(ground_truths.shape[0]).cuda()
            
            logit_predictions = model(batch_images)


            total_loss = nn.CrossEntropyLoss()(logit_predictions, ground_truths)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            t0 += (time.time() - t1)

            if self.global_step % self.print_steps == 0:
                message = "Epoch: %d Step: %d LR: %.6f Total Loss: %.4f Runtime: %.2f s/%d iters." % (epoch+1, self.global_step, lr_scheduler.get_last_lr()[-1], total_loss, t0, self.current_step)
                print("==> %s" % (message))
                self.current_step = 0
                t0 = 0.0

        return total_loss




    def val_one_epoch(self, data_loader, model, epoch):
        with torch.no_grad():
            model.eval()

            for i, inputs in enumerate(data_loader):
                images = inputs["image"].cuda()
                
                ground_truths = inputs["label"]
                ground_truths = ground_truths.reshape(ground_truths.shape[0]).cuda()
                
                predictions = model(images)
                
                if i == 0:
                    all_predictions = predictions
                    all_groundtruths = ground_truths
                else:
                    all_predictions = torch.cat((all_predictions, predictions), dim=0)
                    all_groundtruths = torch.cat((all_groundtruths, ground_truths), dim=0)

            total_loss = nn.CrossEntropyLoss()(all_predictions, all_groundtruths)


        print("==> Epoch: %d Loss %.6f ." % (epoch+1, total_loss.cpu().numpy() ))
        torch.cuda.empty_cache()

        return total_loss

    def train(self,model):
        print("==> Create model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model= nn.DataParallel(model)
        model.to(device)

        model.cuda()
        print("==> List learnable parameters")

        print("==> Load data")
        print(len(self.train_image_files))
        print(len(self.validation_image_files))

        train_data_loader = self._dataloader(self.train_image_files, self.train_labels, split='train', img_transform=self.img_transform)
        val_data_loader = self._dataloader(self.validation_image_files, self.validation_labels, split='val', img_transform=self.img_transform)

        print("==> Configure optimizer")
        optimizer, lr_scheduler = configure_optimizers(model, self.init_lr, self.weight_decay,
                                                       self.gamma, self.lr_decay_every_x_epochs)

        print("==> Start training")
        since = time.time()

        loss_list = []
        
        print('==> Create the saving dictionary')
        if os.path.exists(self.save_checkpoints_dir):
            print('The directory exists, overrode duplicate files')
        else:
            print('Created new dictionary for saving checkpoints')
            os.makedirs(self.save_checkpoints_dir)
        for epoch in range(self.epoch):

            self.train_one_epoch( epoch, train_data_loader, model, optimizer, lr_scheduler)
            _loss = self.val_one_epoch(val_data_loader, model, epoch)
            loss_list.append(_loss.detach().cpu().numpy())

            bestcheckpointstr = "/checkpoint_best_"+ str(self.date.year) + str(self.date.month) + str(self.date.day) + str(self.date.hour) + ".ckpt"
            
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),}, self.save_checkpoints_dir+'/checkpoint_'+str(epoch)+'_iteration.ckpt')
            if _loss.detach().cpu().numpy()<= min(loss_list):
                torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),}, self.save_checkpoints_dir+bestcheckpointstr)
            lr_scheduler.step()
            

        
        print("==> Runtime: %.2f minutes." % ((time.time()-since)/60.0))
        return model