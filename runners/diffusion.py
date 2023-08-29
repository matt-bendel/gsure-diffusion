import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from functions.denoising import efficient_generalized_steps
from mri_utils import ksp_to_viewable_image, FFT_Wrapper, FFT_NN_Wrapper
from core.parser import init_obj
from data_loaders.CondMRIDataModule import CondMRIDataModule
import torchvision.utils as tvu

import random

def data_transform(config, X):
    return X / 7e-5


def inverse_data_transform(config, X):
    return X * 7e-5

def define_network(logger, opt, network_opt):
    """ define network with weights initialization """
    net = init_obj(network_opt, logger)

    if opt['phase'] == 'train':
        logger.info('Network [{}] weights initialize using [{:s}] method.'.format(net.__class__.__name__,
                                                                                  network_opt['args'].get('init_type',
                                                                                                          'default')))
        # net.init_weights()
    return net

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = 'fixedsmall'
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, opt, phase_logger):
        cls_fn = None
        model = define_network(phase_logger, opt, opt['model']['network'])
        state_dict = torch.load(self.args.model_path)
        if 'module.temb.dense.0.weight' in state_dict:
            # model saved as DDP
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        if 'model.temb.dense.0.weight' in state_dict:
            # wrapper saved instead of model
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        print(model.load_state_dict(state_dict))
        if opt['model']['model_wrapper'] if 'model_wrapper' in opt['model'] else False:
            model = FFT_NN_Wrapper(model)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.eval()

        model = Model(self.config)

        self.sample_sequence(model, cls_fn)

    def sample_sequence(self, model, cls_fn=None):
        args, config = self.args, self.config

        #get original images and corrupted y_0
        args.subset_start = 0
        args.subset_end = 72

        dm = CondMRIDataModule()
        dm.setup()
        val_loader = dm.test_dataloader()
        

        ## get degradation matrix ##
        H_funcs = None

        from functions.svd_replacement import Inpainting
        m = np.zeros((resolution, resolution))

        a = [1, 23, 42, 60, 77, 92, 105, 117, 128, 138, 147, 155, 162, 169, 176, 182, 184, 185, 186, 187, 188, 189, 190,
             191, 192, 193, 194, 195,
             196, 197, 198, 199, 200, 204, 210, 217, 224, 231, 239, 248, 258, 269, 281, 294, 309, 326, 344, 363]

        a = np.array(a)
        m[:, a] = 1

        mask = torch.from_numpy(m).to(self.device).reshape(-1)
        missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 2
        missing_c = missing_r + 1

        missing = torch.cat([missing_r, missing_c], dim=0)
        H_funcs = Inpainting(2, 384, missing, self.device)

        args.sigma_0 = 2 * args.sigma_0 #to account for scaling to [-1,1]
        sigma_0 = args.sigma_0
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y_0 = H_funcs.H(x_orig)
            y_0 = y_0 + sigma_0 * torch.randn_like(y_0)

            pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], 2, 384, 384)
            pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1

            for i in range(len(pinv_y_0)):
                tvu.save_image(
                    inverse_data_transform(config, pinv_y_0[i]), os.path.join(self.args.image_folder, f"y0_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]), os.path.join(self.args.image_folder, f"orig_{idx_so_far + i}.png")
                )

            ##Begin DDIM
            x = torch.randn(
                y_0.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
            with torch.no_grad():
                x, _ = self.sample_image(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)

            x = [inverse_data_transform(config, y) for y in x]

            for i in [-1]: #range(len(x)):
                for j in range(x[i].size(0)):
                    tvu.save_image(
                        x[i][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{i}.png")
                    )
                    if i == len(x)-1 or i == -1:
                        orig = inverse_data_transform(config, x_orig[j])
                        mse = torch.mean((x[i][j].to(self.device) - orig) ** 2)
                        psnr = 10 * torch.log10(1 / mse)
                        avg_psnr += psnr

            idx_so_far += y_0.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))

    def sample_image(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
            etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta, cls_fn=cls_fn, classes=classes)
        if last:
            x = x[0][-1]
        return x