import os
import time
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from config import cfg
from utils import weights_init, load_params, copy_G_params
from model import BERT_ENCODER, D_NET32, D_NET64, D_NET128, G_NET
from losses import discriminator_loss, generator_loss, KL_loss
from datasets import prepare_data

# ################# Text to image task############################ #
class BertAttnGANTrainer(object):
    def __init__(self, data_loader):
        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def build_models(self):
        # ###################encoders######################################## #
        text_encoder = BERT_ENCODER()
        for p in text_encoder.parameters():
           p.requires_grad = False
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = [D_NET32(), D_NET64(), D_NET128()]
        netG = G_NET()
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)

        epoch = 0
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [text_encoder, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
            optimizersD.append(opt)
        optimizerG = optim.Adam(netG.parameters(), lr=cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))
        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()
        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(), 'netG_epoch_%d.pth' % (epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(), 'netD%d.pth' % (i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def train(self):
        text_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        segs_ids = Variable(torch.LongTensor(25).fill_(0))
        if cfg.CUDA:
            noise = noise.cuda()
            fixed_noise = fixed_noise.cuda()
            segs_ids = segs_ids.cuda()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, caps, attn_masks, keys = prepare_data(data)

                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(caps, attn_masks, segs_ids)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (caps == 0)

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i], sent_emb, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.data.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs = generator_loss(netsD, fake_imgs, real_labels, sent_emb)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.data.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch + 1, self.max_epoch, self.num_batches,
                     errD_total.data.item(), errG_total.data.item(),
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)
