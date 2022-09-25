import numpy as np
import random
import torch
import argparse
import pathlib
from utils import logs
import logging
import shutil
import sys
import datetime
from torch.utils.tensorboard import SummaryWriter
from models.cInvNet import cInvNet
from models.cInvNet_MRI_ConditionalPrior import cInvNet_MRI_ConditionalPrior
from models.cInvNet_ConditionalPrior import cInvNet_ConditionalPrior
from models.cInvNet_MRI_LearnablePrior import cInvNet_MRI_LearnablePrior
from models.MUNet import MUNet
from loader.dataloader import twoD_Data, RandomDownscale_function_v2, RandomFlip_function, RandomShift_function, Met_Sampler
import torchvision
import matplotlib.pyplot as plt
from torch.nn import functional as F
from pytorch_msssim import ms_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
loss_fn_alex = lpips.LPIPS(net='alex').cuda()
from utils import utils


def DC_loss(met_LR, output, lowRes):
    met_LR_cplx = torch.cat((met_LR.unsqueeze(-1), torch.zeros_like(met_LR.unsqueeze(-1))), -1)
    subkspace = utils.fft2(met_LR_cplx)
    output_cplx = torch.cat((output.unsqueeze(-1), torch.zeros_like(output.unsqueeze(-1))), -1)
    output_kspace = utils.fft2(output_cplx)
    d = output.shape[-1] // 2
    div = lowRes
    mask = torch.zeros_like(output_kspace)
    mask[:, :, d - div:d + div, d - div:d + div, :] = 1.0
    subkspace = subkspace * mask
    output_kspace = output_kspace * mask
    return F.l1_loss(subkspace, output_kspace)


def reshape_noise(args, z_flat, met_HR):
    sizes = [(2 ** i, met_HR.shape[3] // 2 ** i, met_HR.shape[3] // 2 ** i) for i in range(1, args.down_num)]
    sizes.append((2 ** (args.down_num - 1) * 4, met_HR.shape[3] // 2 ** args.down_num, met_HR.shape[3] // 2 ** args.down_num))
    z_sample = []
    cur_dim = 0
    for z_shape in sizes:
        z_dim = np.prod(z_shape)
        this_z = z_flat[:, cur_dim: cur_dim + z_dim]
        this_z = this_z.view(z_flat.size(0), *z_shape)
        z_sample.append(this_z)
        cur_dim += z_dim
    return z_sample


def ssimloss(output, target):
    output_ = F.interpolate(output, mode='bicubic', size=(192, 192), align_corners=True)
    target_ = F.interpolate(target, mode='bicubic', size=(192, 192), align_corners=True)
    ssim_loss = 1 - ms_ssim(output_, target_, data_range=torch.max(target_) - torch.min(target_), size_average=True)
    return ssim_loss


def create_data_loaders(args):
    train_Dataset = twoD_Data(patients=args.train_patients, transform=None, mode='train')
    met_sampler = Met_Sampler(batch_size=args.batch_size, length=train_Dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_Dataset, batch_sampler=met_sampler, num_workers=args.num_workers)
    valid_Dataset = twoD_Data(patients=args.valid_patients, transform=None, mode='valid')
    valid_loader = torch.utils.data.DataLoader(valid_Dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_Dataset = twoD_Data(patients=args.test_patients, transform=None, mode='test')
    test_loader = torch.utils.data.DataLoader(test_Dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    display_Dataset = [train_Dataset[i] for i in range(0, len(train_Dataset), len(train_Dataset) // 100)]
    display_loader = torch.utils.data.DataLoader(display_Dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, valid_loader, test_loader, display_loader


def save_python_script(args):
    ## copy training
    shutil.copyfile(sys.argv[0], str(args.exp_dir) + '/' + sys.argv[0])
    ## copy model
    shutil.copyfile('models/' + args.model + '.py', str(args.exp_dir) + '/' + args.model + '.py')
    ## copy loader
    shutil.copyfile('loader/dataloader.py', str(args.exp_dir) + '/dataloader.py')


def save_model(args, exp_dir, epoch, model, optimizer, valid_loss, best_valid_loss):
    logging.info('Saving trained model')

    ## create a models folder if not exists
    if not (args.exp_dir / 'models').exists():
        (args.exp_dir / 'models').mkdir(parents=True, exist_ok=True)

    if epoch % 100 == 99 or epoch == 0:
        torch.save({'epoch': epoch, 'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   f=str(exp_dir) + '/models/epoch' + str(epoch) + '.pt')
        logging.info('Done saving model')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save({'epoch': epoch, 'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   f=str(exp_dir) + '/models/best_model.pt')
        logging.info('Done saving best model')
    return best_valid_loss


def build_model(args):
    if args.model == 'cInvNet':
        model = cInvNet(channel_in=1, block_num=args.block_num, feature=args.feature, down_num=args.down_num).cuda()
    elif args.model == 'cInvNet_MRI_ConditionalPrior':
        model = cInvNet_MRI_ConditionalPrior(channel_in=1, block_num=args.block_num, feature=args.feature, down_num=args.down_num).cuda()
    elif args.model == 'cInvNet_ConditionalPrior':
        model = cInvNet_ConditionalPrior(channel_in=1, block_num=args.block_num, feature=args.feature, down_num=args.down_num).cuda()
    elif args.model == 'cInvNet_MRI_LearnablePrior':
        model = cInvNet_MRI_LearnablePrior(channel_in=1, block_num=args.block_num, feature=args.feature, down_num=args.down_num).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    SRmodel = MUNet(in_channels=2, init_features=8, latent_dim=32).cuda()
    return model, optimizer, SRmodel


def train_epoch(args, epoch, model, SRmodel, data_loader, optimizer, writer, global_step):
    model.train()
    SRmodel.eval()
    running_loss_nll, running_loss_logpz, running_loss_logdet, running_loss_L1, running_loss_ssim, running_loss_DC = 0, 0, 0, 0, 0, 0
    total_data = len(data_loader)
    for iter, data in enumerate(data_loader):
        T1, flair, met_HR, data_max, Patient, sli, metname = data
        T1 = T1.float().cuda()
        flair = flair.float().cuda()
        met_HR = met_HR.float().cuda()

        if epoch == 0 and iter == 0:
            logging.info('--+' * 10)
            logging.info(f'T1 = {T1.shape}')
            logging.info(f'flair = {flair.shape}')
            logging.info(f'met_HR = {met_HR.shape} ')
            logging.info('--+' * 10)

        T1, flair, met_HR = RandomFlip_function(T1, flair, met_HR)
        T1, flair, met_HR = RandomShift_function(T1, flair, met_HR)
        met_LR, lowRes = RandomDownscale_function_v2(met_HR, args.low_resolution)

        with torch.no_grad():
            SR_pre = SRmodel(T1, flair, met_LR, lowRes)
            nonzero_mask = (met_HR != 0).float()
            SR_pre = torch.mul(SR_pre, nonzero_mask)

        if args.NoEnhancer:
            SR_pre = torch.mul(met_LR, nonzero_mask)
            #print('yes')

        if global_step == 0:
            with torch.no_grad():
                z, logdet = model(met_HR, SR_pre, T1, flair, lowRes, metname, initialize=True, cal_jacobian=True)
            global_step += 1
            continue

        z, logdet = model(met_HR, SR_pre, T1, flair, lowRes, metname, cal_jacobian=True)
        logdet += float(-np.log(256.) * met_HR.shape[1]*met_HR.shape[2]*met_HR.shape[3])
        logdet = logdet / float(np.log(2.)*met_HR.shape[1]*met_HR.shape[2]*met_HR.shape[3])
        logdet = -logdet.mean()

        z_ = [z_.view(z_.shape[0], -1) for z_ in z]
        z = torch.cat(z_, dim=1)
        prior_mean, prior_std = model.prior(SR_pre, T1, flair, metname)
        #print(prior_mean[0, :10], prior_std[0, :10])
        logpz = -0.5 * ((((z - prior_mean) / prior_std) ** 2) + torch.log(2 * np.pi * prior_std ** 2)).sum(-1)
        logpz = logpz / float(np.log(2.)*met_HR.shape[1]*met_HR.shape[2]*met_HR.shape[3])
        logpz = -logpz.mean()

        nll = logdet + logpz

        #guide loss
        loss_L1, loss_ssim = torch.tensor(0.0), torch.tensor(0.0)
        z_flat_guide = prior_mean
        if args.guide_weight > 0.0:
            z_sample_guide = reshape_noise(args, z_flat_guide, met_HR)
            outputs_back_guide = model(z_sample_guide, SR_pre, T1, flair, lowRes, metname, rev=True)
            outputs_back_guide = torch.mul(outputs_back_guide, nonzero_mask)
            loss_L1 = F.l1_loss(outputs_back_guide, met_HR)
            loss_ssim = ssimloss(outputs_back_guide, met_HR)

        #DC loss
        loss_DC = torch.tensor(0.0)
        gaussian_scale = torch.rand(1).cuda()
        z_flat_DC = prior_mean + gaussian_scale * prior_std * torch.randn_like(prior_mean)
        if args.DC_weight > 0.0:
            z_sample_DC = reshape_noise(args, z_flat_DC, met_HR)
            outputs_back_DC = model(z_sample_DC, SR_pre, T1, flair, lowRes, metname, rev=True)
            outputs_back_DC = torch.mul(outputs_back_DC, nonzero_mask)
            loss_DC = DC_loss(met_LR, outputs_back_DC, lowRes)

        loss = args.nll_weight * nll + args.guide_weight * (args.L1_weight * loss_L1 + args.SSIM_weight * loss_ssim) + args.DC_weight * loss_DC
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss_nll += nll.item()
        running_loss_logpz += logpz.item()
        running_loss_logdet += logdet.item()
        running_loss_L1 += loss_L1.item()
        running_loss_ssim += loss_ssim.item()
        running_loss_DC += loss_DC.item()

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{total_data:4d}] '
                f'NLL Loss = {nll.item():.5g} '
                f'Logpz Loss = {logpz.item():.5g} '
                f'Logdet Loss = {logdet.item():.5g} '
                f'L1 Loss = {loss_L1.item():.5g} '
                f'SSIM Loss = {loss_ssim.item():.5g} '
                f'DC Loss = {loss_DC.item():.5g} '
            )
        global_step += 1

    nll = running_loss_nll / total_data
    logpz = running_loss_logpz / total_data
    logdet = running_loss_logdet / total_data
    loss_DC = running_loss_DC / total_data
    loss_L1 = running_loss_L1 / total_data
    loss_ssim = running_loss_ssim / total_data
    if writer is not None:
        writer.add_scalar('Train/NLL', nll, epoch)
        writer.add_scalar('Train/Logpz', logpz, epoch)
        writer.add_scalar('Train/Logdet', logdet, epoch)
        writer.add_scalar('Train/DC', loss_DC, epoch)
        writer.add_scalar('Train/L1', loss_L1, epoch)
        writer.add_scalar('Train/SSIM', loss_ssim, epoch)
    return loss, global_step


def valid(args, epoch, model, SRmodel, data_loader, writer):
    model.eval()
    SRmodel.eval()
    running_loss_nll, running_loss_logpz, running_loss_logdet, running_loss_L1, running_loss_ssim, running_loss_DC, running_loss = 0, 0, 0, 0, 0, 0, 0
    total_data = len(data_loader)
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            T1, flair, met_HR, data_max, Patient, sli, metname = data
            T1 = T1.float().cuda()
            flair = flair.float().cuda()
            met_HR = met_HR.float().cuda()

            T1, flair, met_HR = RandomFlip_function(T1, flair, met_HR)
            T1, flair, met_HR = RandomShift_function(T1, flair, met_HR)
            met_LR, lowRes = RandomDownscale_function_v2(met_HR, args.low_resolution)

            SR_pre = SRmodel(T1, flair, met_LR, lowRes)
            nonzero_mask = (met_HR != 0).float()
            SR_pre = torch.mul(SR_pre, nonzero_mask)

            if args.NoEnhancer:
                SR_pre = torch.mul(met_LR, nonzero_mask)

            z, logdet = model(met_HR, SR_pre, T1, flair, lowRes, metname, cal_jacobian=True)
            logdet += float(-np.log(256.) * met_HR.shape[1] * met_HR.shape[2] * met_HR.shape[3])
            logdet = logdet / float(np.log(2.) * met_HR.shape[1] * met_HR.shape[2] * met_HR.shape[3])
            logdet = -logdet.mean()

            z_ = [z_.view(z_.shape[0], -1) for z_ in z]
            z = torch.cat(z_, dim=1)
            prior_mean, prior_std = model.prior(SR_pre, T1, flair, metname)
            logpz = -0.5 * ((((z - prior_mean) / prior_std) ** 2) + torch.log(2 * np.pi * prior_std ** 2)).sum(-1)
            logpz = logpz / float(np.log(2.) * met_HR.shape[1] * met_HR.shape[2] * met_HR.shape[3])
            logpz = -logpz.mean()

            nll = logdet + logpz

            # guide loss
            loss_L1, loss_ssim = torch.tensor(0.0), torch.tensor(0.0)
            z_flat_guide = prior_mean
            if args.guide_weight > 0.0:
                z_sample_guide = reshape_noise(args, z_flat_guide, met_HR)
                outputs_back_guide = model(z_sample_guide, SR_pre, T1, flair, lowRes, metname, rev=True)
                outputs_back_guide = torch.mul(outputs_back_guide, nonzero_mask)
                loss_L1 = F.l1_loss(outputs_back_guide, met_HR)
                loss_ssim = ssimloss(outputs_back_guide, met_HR)

            # DC loss
            loss_DC = torch.tensor(0.0)
            gaussian_scale = torch.rand(1).cuda()
            z_flat_DC = prior_mean + gaussian_scale * prior_std * torch.randn_like(prior_mean)
            if args.DC_weight > 0.0:
                z_sample_DC = reshape_noise(args, z_flat_DC, met_HR)
                outputs_back_DC = model(z_sample_DC, SR_pre, T1, flair, lowRes, metname, rev=True)
                outputs_back_DC = torch.mul(outputs_back_DC, nonzero_mask)
                loss_DC = DC_loss(met_LR, outputs_back_DC, lowRes)

            loss = args.nll_weight * nll + args.guide_weight * (args.L1_weight * loss_L1 + args.SSIM_weight * loss_ssim) + args.DC_weight * loss_DC

            running_loss_nll += nll.item()
            running_loss_logpz += logpz.item()
            running_loss_logdet += logdet.item()
            running_loss_L1 += loss_L1.item()
            running_loss_ssim += loss_ssim.item()
            running_loss_DC += loss_DC.item()
            running_loss += loss.item()

        nll = running_loss_nll / total_data
        logpz = running_loss_logpz / total_data
        logdet = running_loss_logdet / total_data
        loss_L1 = running_loss_L1 / total_data
        loss_ssim = running_loss_ssim / total_data
        loss_DC = running_loss_DC / total_data
        loss = running_loss / total_data

    if writer is not None:
        writer.add_scalar('Dev/NLL', nll, epoch)
        writer.add_scalar('Dev/Logpz', logpz, epoch)
        writer.add_scalar('Dev/Logdet', logdet, epoch)
        writer.add_scalar('Dev/L1', loss_L1, epoch)
        writer.add_scalar('Dev/SSIM', loss_ssim, epoch)
        writer.add_scalar('Dev/DC', loss_DC, epoch)
        writer.add_scalar('Dev/Loss_Total', loss, epoch)
    return loss


def visualize(args, epoch, model, SRmodel, data_loader, writer):
    def save_image(image, tag):
        image -= image.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
        image /= image.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    SRmodel.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            T1, flair, met_HR, data_max, Patient, sli, metname = data
            T1 = T1.float().cuda()
            flair = flair.float().cuda()
            met_HR = met_HR.float().cuda()

            T1, flair, met_HR = RandomFlip_function(T1, flair, met_HR)
            T1, flair, met_HR = RandomShift_function(T1, flair, met_HR)
            met_LR, lowRes = RandomDownscale_function_v2(met_HR, args.low_resolution)

            SR_pre = SRmodel(T1, flair, met_LR, lowRes)
            nonzero_mask = (met_HR != 0).float()
            SR_pre = torch.mul(SR_pre, nonzero_mask)

            if args.NoEnhancer:
                SR_pre = torch.mul(met_LR, nonzero_mask)

            prior_mean, prior_std = model.prior(SR_pre, T1, flair, metname)
            z_flat = prior_mean + args.gaussian_scale * prior_std * torch.randn_like(prior_mean)
            z_sample = reshape_noise(args, z_flat, met_HR)

            outputs = model(z_sample, SR_pre, T1, flair, lowRes, metname, rev=True)
            nonzero_mask = (met_HR != 0).float()
            outputs = torch.mul(outputs, nonzero_mask)

            save_image(met_HR, 'Target')
            save_image(outputs, 'Output')
            save_image(met_LR, 'Input_Met')
            save_image(SR_pre, 'MUNet_SR')
            save_image(T1[:, :, :, :, 1], 'Input_T1')
            #save_image(flair[:, :, :, :, 1], 'Input_flair')
            break


def main_train(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    logs.set_logger(str(args.exp_dir / 'train.log'))
    logging.info('--' * 10)
    logging.info(
        '%s create log file %s' % (datetime.datetime.now().replace(microsecond=0), str(args.exp_dir / 'train.log')))

    save_python_script(args)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    model, optimizer, SRmodel = build_model(args)
    SRmodel.load_state_dict(torch.load('checkpoints/NormalizingFlow/newdata/MUNet_lr8/testpatients_1_2_4_6_8/models/best_model.pt')['model']) #path to pretrained MUNet

    logging.info('--' * 10)
    logging.info(args)
    logging.info('--' * 10)
    logging.info(model)
    logging.info('--' * 10)
    logging.info('Total parameters: %s' % sum(p.numel() for p in model.parameters()))
    logging.info('--' * 10)

    train_loader, valid_loader, test_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    logging.info('--' * 10)
    start_training = datetime.datetime.now().replace(microsecond=0)
    logging.info('Start training at %s' % str(start_training))

    best_valid_loss = 1e9
    valid_loss = 1e10
    global_step = 0
    for epoch in range(0, args.num_epochs):
        logging.info('Current LR %s' % (scheduler.get_lr()[0]))
        torch.manual_seed(args.seed + epoch)
        train_loss, global_step = train_epoch(args, epoch, model, SRmodel, train_loader, optimizer, writer, global_step)

        if epoch > args.num_epochs * args.valid_epoch:
            valid_loss = valid(args, epoch, model, SRmodel, valid_loader, writer)
        best_valid_loss = save_model(args, args.exp_dir, epoch, model, optimizer, valid_loss, best_valid_loss)
        visualize(args, epoch, model, SRmodel, display_loader, writer)

        scheduler.step(epoch)
        logging.info('Epoch: %s Reduce LR to: %s' % (epoch, scheduler.get_lr()[0]))

    writer.close()


def compute_LPIPS(img, GT, loss):
    img = torch.from_numpy(img).float().cuda()
    img = img.unsqueeze(0).unsqueeze(0)
    img = img.expand(1, 3, 64, 64)
    img = (img * 2 - img.max()) / img.max()

    GT = torch.from_numpy(GT).float().cuda()
    GT = GT.unsqueeze(0).unsqueeze(0)
    GT = GT.expand(1, 3, 64, 64)
    GT = (GT * 2 - GT.max()) / GT.max()
    LPIPS = loss(img, GT).data
    LPIPS = LPIPS.cpu().numpy()[0][0][0][0]
    return LPIPS


def main_evaluate(args):
    save_dir = (args.exp_dir / ('results_temp' + str(args.gaussian_scale)))
    save_dir.mkdir(parents=True, exist_ok=True)
    logs.set_logger(str(args.exp_dir / ('eval_temp' + str(args.gaussian_scale) + '.log')))
    logging.info('--' * 10)
    logging.info('%s create log file %s' % (datetime.datetime.now().replace(microsecond=0), str(args.exp_dir / ('eval_temp' + str(args.gaussian_scale) + '.log'))))

    train_loader, valid_loader, test_loader, display_loader = create_data_loaders(args)
    del train_loader, valid_loader, display_loader

    logging.info('--' * 10)
    start_eval = datetime.datetime.now().replace(microsecond=0)
    logging.info('Loading model %s' % str(args.checkpoint))
    logging.info('Start Evaluation at %s' % str(start_eval))
    logging.info('low resolution = %s' % str(args.low_resolution))

    model, optimizer, SRmodel = build_model(args)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    SRmodel.load_state_dict(torch.load('checkpoints/NormalizingFlow/newdata/MUNet_lr8/testpatients_1_2_4_6_8/models/best_model.pt')['model']) #path to pretrained MUNet
    model.eval()
    SRmodel.eval()
    running_psnr, running_ssim, running_lpips = [], [], []
    total_data = len(test_loader)
    with torch.no_grad():
        for iter, data in enumerate(test_loader):
            T1, flair, met_HR, data_max, Patient, sli, metname = data
            T1 = T1.float().cuda()
            flair = flair.float().cuda()
            met_HR = met_HR.float().cuda()

            met_LR, lowRes = RandomDownscale_function_v2(met_HR, args.low_resolution)

            SR_pre = SRmodel(T1, flair, met_LR, lowRes)
            nonzero_mask = (met_HR != 0).float()
            SR_pre = torch.mul(SR_pre, nonzero_mask)

            if args.NoEnhancer:
                SR_pre = torch.mul(met_LR, nonzero_mask)

            prior_mean, prior_std = model.prior(SR_pre, T1, flair, metname)
            print(prior_mean.mean(), prior_std.mean())
            z_flat = prior_mean + args.gaussian_scale * prior_std * torch.randn_like(prior_mean)
            z_sample = reshape_noise(args, z_flat, met_HR)

            output = model(z_sample, SR_pre, T1, flair, lowRes, metname, rev=True)
            output = torch.mul(output, nonzero_mask)
            output = torch.clamp(output, min=0.0, max=1.0)
            #output = SR_pre

            data_max = data_max.numpy()
            output = output.squeeze().cpu().numpy() * data_max
            met_HR = met_HR.squeeze().cpu().numpy() * data_max
            met_LR = met_LR.squeeze().cpu().numpy() * data_max
            T1 = T1.squeeze().cpu().numpy()
            flair = flair.squeeze().cpu().numpy()

            PSNR = peak_signal_noise_ratio(met_HR, output, data_range=met_HR.max() - met_HR.min())
            SSIM = structural_similarity(met_HR, output, data_range=met_HR.max() - met_HR.min())
            LPIPS = compute_LPIPS(output, met_HR, loss_fn_alex)

            running_psnr.append(PSNR.item())
            running_ssim.append(SSIM.item())
            running_lpips.append(LPIPS.item())

            sli = sli.numpy()
            logging.info(f'{Patient[0]} slice={sli[0]} met={metname[0]} PSNR={PSNR} SSIM={SSIM} LPIPS={LPIPS}')

            if metname[0] == 'Gln':
                output_file = '%s/%s_slice%s_MRI.npz' % (str(save_dir), Patient[0], sli[0])
                np.savez(output_file, T1=T1, flair=flair)
            output_file = '%s/%s_slice%s_%s.npz' % (str(save_dir), Patient[0], sli[0], metname[0])
            np.savez(output_file, met_LR=met_LR, output=output, met_HR=met_HR)

            met_LR_plot = met_LR.repeat(3, axis=0).repeat(3, axis=1)
            met_HR_plot = met_HR.repeat(3, axis=0).repeat(3, axis=1)
            outputs_plot = output.repeat(3, axis=0).repeat(3, axis=1)
            output_file = '%s/%s_slice%s_T1.png' % (str(save_dir), Patient[0], sli[0])
            plt.imsave(output_file, np.concatenate((T1[:, :, 0], T1[:, :, 1], T1[:, :, 2]), axis=1), cmap='gray')
            output_file = '%s/%s_slice%s_flair.png' % (str(save_dir), Patient[0], sli[0])
            plt.imsave(output_file, np.concatenate((flair[:, :, 0], flair[:, :, 1], flair[:, :, 2]), axis=1), cmap='gray')
            output_file = '%s/%s_slice%s_%s.png' % (str(save_dir), Patient[0], sli[0], metname[0])
            plt.imsave(output_file, np.concatenate((met_LR_plot, met_HR_plot, outputs_plot), axis=1), cmap='jet', vmin=met_HR_plot.min(), vmax=met_HR_plot.max())

    running_psnr = np.asarray(running_psnr)
    running_ssim = np.asarray(running_ssim)
    running_lpips = np.asarray(running_lpips)
    logging.info('PSNR = %5g +- %5g' % (running_psnr.mean(), running_psnr.std()))
    logging.info('SSIM = %5g +- %5g' % (running_ssim.mean(), running_ssim.std()))
    logging.info('LPIPS = %5g +- %5g' % (running_lpips.mean(), running_lpips.std()))
    np.savez(str(args.exp_dir) + '/metrics_temp' + str(args.gaussian_scale) + '.npz', psnr=running_psnr, ssim=running_ssim, lpips=running_lpips)


def create_arg_parser():
    parser = argparse.ArgumentParser()

    ## train
    parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
    parser.add_argument('--lr-step-size', type=int, default=70, help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--num-epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--valid-epoch', type=float, default=0.8, help='number of epochs to start validation')
    parser.add_argument('--report-interval', type=int, default=10, help='Period of printing loss')
    parser.add_argument('--batch-size', default=8, type=int, help='batch size')
    parser.add_argument('--num-workers', default=0, type=int, help='number of works')
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--L1-weight', type=float, default=0.16, help='weight of L1 loss backward')
    parser.add_argument('--SSIM-weight', type=float, default=0.84, help='weight of SSIM loss backward')
    parser.add_argument('--DC-weight', type=float, default=10.0, help='weight of DC loss')
    parser.add_argument('--nll-weight', type=float, default=1.0, help='weight of NLL loss')
    parser.add_argument('--guide-weight', type=float, default=10.0, help='weight of guidance loss')
    parser.add_argument('--low-resolution', type=int, default=8, help='half of the low resolution matrix size')

    ## methods
    parser.add_argument('--model', type=str, default='cInvNet_MRI_ConditionalPrior', choices=['cInvNet', 'cInvNet_ConditionalPrior',
                                                                                              'cInvNet_MRI_LearnablePrior', 'cInvNet_MRI_ConditionalPrior'], help='model')
    parser.add_argument('--block-num', nargs='+', type=int, default=[12, 12, 12, 12], help='number of blocks for each downscaling block')
    parser.add_argument('--down-num', type=int, default=3, help='number of downsampling blocks')
    parser.add_argument('--feature', type=int, default=128, help='latent dimension for conditional networks')
    parser.add_argument('--gaussian_scale', type=float, default=0.8, help='gaussian noise scale')
    parser.add_argument('--NoEnhancer', action='store_true', help='If set, input LR instead of SRPre')
    parser.add_argument('--test-patients', nargs='+', type=int, default=[1, 2, 4, 6, 8], help='Patient # for testing')
    parser.add_argument('--valid-patients', nargs='+', type=int, default=[9, 12, 14, 15, 16], help='Patient # for validation')
    parser.add_argument('--train-patients', nargs='+', type=int, default=[17, 18, 19, 20, 22, 73, 78, 81, 84, 86, 91, 93, 96, 100, 104], help='Patient # for training')
    parser.add_argument('--exp-dir', type=str, default='checkpoints', help='Path to save models and results')

    ## evaluation
    parser.add_argument('--evaluate', action='store_true', help='If set, evaluation mode')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='path where the model was saved')
    args = parser.parse_args()

    args.exp_dir = args.exp_dir + '/' + args.model + '_blocknum' + str(args.block_num[0])+ '_downnum' + str(args.down_num) \
                   + '_features' + str(args.feature) + '_DCweight' + str(args.DC_weight) + '_guideweight' + str(args.guide_weight)
    args.exp_dir = args.exp_dir + '_lr' + str(args.low_resolution)
    if args.NoEnhancer:
        args.exp_dir = args.exp_dir + '_NoEnhancer'
    args.exp_dir = args.exp_dir + '/testpatients_' + str(args.test_patients[0]) + '_' + str(args.test_patients[1]) + \
                   '_' + str(args.test_patients[2]) + '_' + str(args.test_patients[3]) + '_' + str(args.test_patients[4])
    args.checkpoint = args.exp_dir + '/models/' + args.checkpoint
    args.exp_dir = pathlib.Path(args.exp_dir)

    return args


if __name__ == '__main__':
    args = create_arg_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.evaluate:
        print('evaluate')
        print('--' * 10)
        main_evaluate(args)
    else:
        print('train')
        print('--' * 10)
        main_train(args)
