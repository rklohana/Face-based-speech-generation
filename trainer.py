from model_spec import Generator
from model_spec import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime

class Solver(object):
    def __init__(self, spec_loader, config):
        self.spec_loader = spec_loader
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs
        self.test_iters = config.test_iters
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        self.build_model()

    def build_model(self):
        if self.dataset in ['CelebA', 'RaFD', 'Spec']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        print("\n")

    def restore_model(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return (dydx_l2norm - 1)**2

    def classification_loss(self, logit, target, dataset='CelebA'):
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)
        elif dataset == "Spec":
            return F.cosine_similarity(logit, target)

    def train(self):
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        elif self.dataset == 'Spec':
            data_loader = self.spec_loader
        print("The length of data_loader is {}".format(len(data_loader)))
        data_iter = iter(data_loader)
        x_id_fixed, c_id_fixed, x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed[:, :, :, :256]
        x_fixed = x_fixed.to(self.device)
        g_lr = self.g_lr
        d_lr = self.d_lr
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            try:
                x_id, c_id, x_real, label_org = next(data_iter)
                x_real = x_real[:, :, :, :-1]
            except:
                data_iter = iter(data_loader)
                x_id, c_id, x_real, label_org = next(data_iter)
                x_real = x_real[:, :, :, :-1]
            print(x_id, c_id)
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)
            elif self.dataset == 'Spec':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            x_real = x_real.to(self.device)
            c_org = c_org.to(self.device)
            c_trg = c_trg.to(self.device)
            label_org = label_org.to(self.device)
            label_trg = label_trg.to(self.device)
            out_src, out_cls = self.D(x_real)
            self.batch_size = int(out_src.shape[0])
            d_loss_real = -torch.mean(out_src, dim=(1,2,3))
            d_loss_real = torch.flatten(d_loss_real)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src, dim=(1,2,3))
            d_loss_fake = torch.flatten(d_loss_fake)
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            d_loss = torch.mean(d_loss)
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            loss = {}
            loss['D/loss_real'] = float(torch.mean(d_loss_real))
            loss['D/loss_fake'] = float(torch.mean(d_loss_fake))
            loss['D/loss_cls'] = float(torch.mean(d_loss_cls))
            loss['D/loss_gp'] = float(torch.mean(d_loss_gp))
            loss['D/loss'] = float(d_loss)
            if (i+1) % self.n_critic == 0:
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = -torch.mean(out_src, dim=(1,2,3))
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst), dim=(1,2,3))
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                g_loss = torch.mean(g_loss)
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()
                loss['G/loss_fake'] = float(torch.mean(g_loss_fake))
                loss['G/loss_rec'] = float(torch.mean(g_loss_rec))
                loss['G/loss_cls'] = float(torch.mean(g_loss_cls))
                loss['G/loss'] = float(g_loss)
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                print("\n")
                print("\n")
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        self.restore_model(self.test_iters)
        if self.dataset == "Spec":
            data_loader = self.spec_loader
        with torch.no_grad():
            for i, (x_id, c_id, x_real, c_shuffled) in enumerate(data_loader):
                x_real = x_real[:, :, :, :-1]
                x_real = x_real.to(self.device)
                c_shuffled = c_shuffled.to(self.device)
                x_fake = self.G(x_real, c_shuffled)
                result_path = os.path.join(self.result_dir, '{}_{}_{}-spec.npy'.format(x_id, c_id, i+1))
                np.save(result_path, x_fake.data.cpu())
                print('Saved real and fake images into {}...'.format(result_path))
