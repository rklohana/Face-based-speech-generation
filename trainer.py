from gan_model import Generator, Discriminator
from dataloader import get_loader
import argparse
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
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.dataset = config.dataset
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.test_iters = config.test_iters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        self.build_model()

    def build_model(self):
        if self.dataset == 'Spec':
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        else:
            raise ValueError("Only the Spec dataset is supported")
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.g_lr, betas=(self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.d_lr, betas=(self.beta1, self.beta2))
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
        self.G.load_state_dict(torch.load(G_path, map_location=self.device, weights_only=True))
        self.D.load_state_dict(torch.load(D_path, map_location=self.device, weights_only=True))

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
        weight = torch.ones_like(y)
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return ((dydx_l2norm - 1)**2).mean()

    def classification_loss(self, logit, target):
        return (1 - F.cosine_similarity(logit, target)).mean()

    def train(self):
        data_loader = self.spec_loader
        print("The length of data_loader is {}".format(len(data_loader)))
        data_iter = iter(data_loader)
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
            except StopIteration:
                data_iter = iter(data_loader)
                x_id, c_id, x_real, label_org = next(data_iter)
            print(x_id, c_id)
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
            c_org = label_org.clone()
            c_trg = label_trg.clone()
            x_real = x_real.to(self.device)
            c_org = c_org.to(self.device)
            c_trg = c_trg.to(self.device)
            label_org = label_org.to(self.device)
            label_trg = label_trg.to(self.device)
            out_src, out_cls = self.D(x_real)
            d_loss_real = -torch.mean(out_src, dim=(1,2,3))
            d_loss_real = torch.flatten(d_loss_real)
            d_loss_cls = self.classification_loss(out_cls, label_org)
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src, dim=(1,2,3))
            d_loss_fake = torch.flatten(d_loss_fake)
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.detach() + (1 - alpha) * x_fake.detach()).requires_grad_(True)
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
                g_loss_cls = self.classification_loss(out_cls, label_trg)
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
        self.G.eval()
        self.D.eval()
        data_loader = self.spec_loader
        with torch.no_grad():
            for i, (x_id, c_id, x_real, c_shuffled) in enumerate(data_loader):
                x_real = x_real.to(self.device)
                c_shuffled = c_shuffled.to(self.device)
                x_fake = self.G(x_real, c_shuffled)
                result_path = os.path.join(self.result_dir, '{}_{}_{}-spec.npy'.format(x_id[0], c_id[0], i+1))
                np.save(result_path, x_fake[0, 0].cpu().numpy())
                print('Saved real and fake images into {}...'.format(result_path))


def main():
    parser = argparse.ArgumentParser(description="Train or test the spectrogram GAN")
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--data_dir", required=True, help="Speaker folders containing .npy spectrograms")
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--ids_path", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--g_conv_dim", type=int, default=64)
    parser.add_argument("--d_conv_dim", type=int, default=64)
    parser.add_argument("--g_repeat_num", type=int, default=6)
    parser.add_argument("--d_repeat_num", type=int, default=6)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_rec", type=float, default=10.0)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--num_iters", type=int, default=200000)
    parser.add_argument("--num_iters_decay", type=int, default=100000)
    parser.add_argument("--g_lr", type=float, default=0.0001)
    parser.add_argument("--d_lr", type=float, default=0.0001)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--resume_iters", type=int, default=0)
    parser.add_argument("--test_iters", type=int, default=0)
    parser.add_argument("--model_save_dir", default="models")
    parser.add_argument("--result_dir", default="results")
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--model_save_step", type=int, default=1000)
    parser.add_argument("--lr_update_step", type=int, default=1000)
    config = parser.parse_args()
    config.dataset = "Spec"
    if config.mode == "test" and not config.test_iters:
        parser.error("--test_iters is required for test mode")
    if min(config.log_step, config.model_save_step, config.lr_update_step, config.n_critic) < 1:
        parser.error("step intervals must be positive")
    if config.num_iters_decay < 1:
        parser.error("--num_iters_decay must be positive")

    loader = get_loader(config.data_dir, config.embeddings_path, config.ids_path,
                        config.image_size, config.batch_size, config.mode, config.num_workers)
    config.c_dim = loader.dataset[0][3].numel()
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)
    solver = Solver(loader, config)
    if config.mode == "train":
        solver.train()
    else:
        solver.test()


if __name__ == "__main__":
    main()
