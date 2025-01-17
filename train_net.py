import os
from statistics import mode
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from util import distributed as du

import time
from collections import OrderedDict
from data import create_dataset
from data import shuffle_dataset
from models import create_model
from util.visualizer import Visualizer
from util.evaluation import evaluation
from util import html,util
from util.visualizer import save_images,save_test_html,save_feature_images

def train(cfg):
    #init
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    #init dataset
    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    model = create_model(cfg)      # create a model given cfg.model and other options
    model.set_position(None,patch_pos=None)
    
    visualizer = Visualizer(cfg)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    # cur_device = torch.cuda.current_device()
    is_master = du.is_master_proc(cfg.NUM_GPUS)
    
    for epoch in range(cfg.epoch_count, cfg.niter + cfg.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        if is_master:
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        shuffle_dataset(dataset, epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            if is_master:
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % cfg.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                    iter_data_time = time.time()
                visualizer.reset()
                total_iters += cfg.batch_size
                epoch_iter += cfg.batch_size
            if epoch == cfg.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(cfg)               # regular setup: load and print networks; create schedulers
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
                

            if total_iters % cfg.print_freq == 0 and is_master:   # display images on visdom and save images to a HTML file
                save_result = total_iters % cfg.update_html_freq == 0
                model.compute_visuals()
                video_path = data['video_object_path'][0].split("/")
                video_name = video_path[-3]+"/"+video_path[-2]
                selected_frames = data['selected_frames'][0]
                selected_frames_str="-".join('%s' %id for id in selected_frames)
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result,dec=video_name+"|"+selected_frames_str)
                
            losses = model.get_current_losses()
            if cfg.NUM_GPUS > 1:
                losses = du.all_reduce(losses)
            if total_iters % cfg.print_freq == 0 and is_master:    # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time)
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if cfg.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            if total_iters % cfg.save_latest_freq == 0 and is_master:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if cfg.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                
        if epoch % cfg.save_epoch_freq == 0 and is_master:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            if cfg.save_iter_model and epoch>=55:
                model.save_networks(epoch)
        if is_master:
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, cfg.niter + cfg.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

        visualizer.time_write(epoch)


def test(cfg):
    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    model = create_model(cfg)      # create a model given cfg.model and other options
    model.set_position(None,patch_pos=None)
    model.setup(cfg)               # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(cfg.results_dir, cfg.name, '%s_%s' % (cfg.phase, cfg.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (cfg.name, cfg.phase, cfg.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if cfg.eval:
        model.eval()
    ismaster = du.is_master_proc(cfg.NUM_GPUS)

    fmse_score_list = []
    mse_scores = 0
    fmse_scores = 0
    comp_mse_scores = 0
    comp_fmse_scores =0
    num_image = 0
    
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.new_test()           # run inference

        visuals = model.get_current_visuals()  # get image results
        
        img_path = model.get_image_paths()     # get image paths # Added by Mia

        if i % 5 == 0 and ismaster:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        visuals_ones = OrderedDict()
        harmonized = None
        real = None
         
        for j in range(len(img_path)):  # len(img_path) = batchsize * t
            img_path_one = []
            for label, im_data in visuals.items():
                bs,t,c,h,w = im_data.shape
                im_data = im_data.reshape(bs*t, c, h, w)
                visuals_ones[label] = im_data[j:j+1, :, :, :]
            img_path_one.append(img_path[j])
            save_images(webpage, visuals_ones, img_path_one, aspect_ratio=cfg.aspect_ratio, width=cfg.display_winsize)
            num_image += 1
            name_parts = img_path[j].split('/')
            raw_name = name_parts[-3]+'/'+name_parts[-2]+'/'+name_parts[-1]
        
            mse_score, fmse_score, comp_mse_score, comp_fmse_score, score_str = evaluation(raw_name, visuals_ones['harmonized']*255, visuals_ones['real']*255, visuals_ones['comp']*255, visuals_ones['mask'])
            fmse_score_list.append(score_str)
            mse_scores += mse_score
            fmse_scores += fmse_score
            comp_mse_scores += comp_mse_score
            comp_fmse_scores += comp_fmse_score
            visuals_ones.clear()
    save_test_html(web_dir, visual_names = model.visual_names, width=cfg.display_winsize)
    mse_mu = mse_scores/num_image
    fmse_mu = fmse_scores/num_image
    comp_mse_mu = comp_mse_scores/num_image
    comp_fmse_mu = comp_fmse_scores/num_image
    mean_score = "%s MSE %0.2f | fMSE %0.2f | comp_MSE %0.2f comp_fMSE %0.2f" % (cfg.dataset_mode, mse_mu, fmse_mu, comp_mse_mu, comp_fmse_mu)
    print(mean_score)
    fmse_score_list = sorted(fmse_score_list, key=lambda image: image[1], reverse=True)
    save_fmse_root = os.path.join(cfg.results_dir, cfg.name)
    save_fmse_path = os.path.join(save_fmse_root,"evaluation_detail_"+cfg.test_epoch+".txt")

    file=open(save_fmse_path,'w') 
    file.write(str(num_image))
    file.write('\n')
    file.write(mean_score)
    file.write('\n')
    # lists=[str(line)+"\n" for line in fmse_score_list]
    for line in fmse_score_list:
        file.write(str(line)+"\n")
    file.close() 


