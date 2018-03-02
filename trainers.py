from os.path import dirname, abspath, join, exists
import os
from datetime import datetime
from tqdm import tqdm
import torch
from torch.autograd import Variable

class Trainer():

    def __init__(self, model, train_dataloader, val_dataloader, criterion, optimizer, lr_schedule, lr_scheduler,
                 use_gpu=False, print_every=1, save_every=1, logger=None):

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion(size_average=False)
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.lr_scheduler = lr_scheduler

        self.print_every = print_every
        self.save_every = save_every

        self.epoch = 0
        self.epoch_losses = []
        self.epoch_metrics = []
        self.val_epoch_losses = []
        self.val_epoch_metrics = []
        self.use_gpu = use_gpu
        self.logger = logger

        self.base_message = ("Epoch: {epoch:<3d} "
                             "Progress: {progress:<.1%} ({elapsed}) "
                             "Train Loss: {train_loss:<.6} "
                             "Train Acc: {train_metric:<.1%} "
                             "Val Loss: {val_loss:<.6} "
                             "Val Acc: {val_metric:<.1%} "
                             "Learning rate: {learning_rate:<.4} "
                            )
        
        self.start_time = datetime.now()

    def train(self):
        self.model.train()
        
        self.batch_losses = []
        self.batch_metrics = []
        for inputs, targets in tqdm(self.train_dataloader):
            
            if self.use_gpu:
                self.inputs, self.targets = Variable(inputs.cuda()), Variable(targets.cuda())
            else:
                self.inputs, self.targets = Variable(inputs), Variable(targets)

            self.optimizer.zero_grad()
            self.outputs = self.model(self.inputs)
            batch_loss = self.criterion(self.outputs, self.targets)
            batch_metric = self.accuracy(self.outputs, self.targets)
                        
            batch_loss.backward()
            self.optimizer.step()

            self.batch_losses.append(batch_loss.data)
            self.batch_metrics.append(batch_metric.data)
            if self.epoch == 0: # for testing
                break

        # validation
        self.model.eval()
        self.val_batch_losses = []
        self.val_batch_metrics = []
        for val_inputs, val_targets in self.val_dataloader:
            if self.use_gpu:
                self.val_inputs, self.val_targets = Variable(val_inputs.cuda()), Variable(val_targets.cuda())
            else:
                self.val_inputs, self.val_targets = Variable(val_inputs), Variable(val_targets)
                
            self.val_outputs = self.model(self.val_inputs)
            val_batch_loss = self.criterion(self.val_outputs, self.val_targets)
            val_batch_metric = self.accuracy(self.val_outputs, self.val_targets)
            self.val_batch_losses.append(val_batch_loss.data)
            self.val_batch_metrics.append(val_batch_metric.data)
        
        train_data_size = len(self.train_dataloader.dataset)
        epoch_loss = torch.cat(self.batch_losses).sum() / train_data_size
        epoch_metric = torch.cat(self.batch_metrics).sum() / train_data_size
        
        val_data_size = len(self.val_dataloader.dataset)
        val_epoch_loss = torch.cat(self.val_batch_losses).sum() / val_data_size
        val_epoch_metric = torch.cat(self.val_batch_metrics).sum() / val_data_size
        
        return epoch_loss, epoch_metric, val_epoch_loss, val_epoch_metric

    def run(self, epochs=10):

        for epoch in range(self.epoch, epochs + 1):
            self.epoch = epoch

            epoch_loss, epoch_metric, val_epoch_loss, val_epoch_metric = self.train()
            if self.lr_schedule:
                self.lr_scheduler.step(val_epoch_loss)
            
            self.epoch_losses.append(epoch_loss)
            self.epoch_metrics.append(epoch_metric)
            self.val_epoch_losses.append(val_epoch_loss)
            self.val_epoch_metrics.append(val_epoch_metric)

            if epoch % self.print_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                message = self.base_message.format(epoch=epoch, 
                                                   progress=epoch/epochs, 
                                                   train_loss=epoch_loss,
                                                   train_metric=epoch_metric,
                                                   val_loss=val_epoch_loss,
                                                   val_metric=val_epoch_metric,
                                                   learning_rate=current_lr,
                                                   elapsed=self.elapsed_time()
                                                  )
                self.logger.info(message)
                
            if epoch % self.save_every == 0:
                self.logger.info("Saving the model...")
                self.save_model()
    
    def accuracy(self, outputs, labels):
        
        maximum, argmax = outputs.max(dim=1)
        corrects = argmax == labels # ByteTensor
        n_corrects = corrects.float().sum() # FloatTensor
        return n_corrects

    def elapsed_time(self):
        now = datetime.now()
        elapsed = now - self.start_time
        return str(elapsed)
    
    def save_model(self):
        base_dir = dirname(abspath(__file__))
        checkpoint_dir = join(base_dir, 'checkpoints')
        if not exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        model_name = self.model.__class__.__name__
        base_filename = '{model_name}-{start_time}-{epoch}.pth'
        checkpoint_filename=base_filename.format(model_name=model_name, 
                                                 start_time=self.start_time,
                                                 epoch=self.epoch)
        checkpoint_filepath = join(checkpoint_dir, checkpoint_filename)
        torch.save(self.model.state_dict(), checkpoint_filepath)
        self.last_checkpoint_filepath = checkpoint_filepath
        if max(self.val_epoch_metrics) == self.val_epoch_metrics[-1]: # if last run is the best
            self.best_checkpoint_filepath = checkpoint_filepath