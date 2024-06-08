import os
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
from ticluv.models.tic_luv import TIC_LUV
import numpy as np
from tensorboardX import SummaryWriter
from datasets import KZDataset
from losses import FocalLoss, MMDLoss


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1
    return float(lr / 2 * cos_out)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=4, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        pred = torch.softmax(pred,dim=1)
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)
        log_p = probs.log()
        alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

    
def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None, imsize=256, num_frame=16, rawh=896, raww=704):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    K = 5
    for ki in range(K):
        use_cuda = torch.cuda.is_available()
        print(use_cuda)

        log_dir = store_name + '/' + 'fold{}'.format(ki)
        writer = SummaryWriter(log_dir)
        # Data
        print('==> Preparing data..')

        trainset = KZDataset(path_0=r'.\data_train\dcm\list0.csv',
                             path_1=r'.\data_train\dcm\list1.csv',
                             path_m=r'.\data_ori\nii',
                             ki=ki, K=K, num_frame=num_frame, image_size=imsize, patch_size_tic=64, rawh=rawh, raww=raww, typ='train', transform=True, rand=True)
        valset = KZDataset(path_0=r'.\data_train\dcm\list0.csv',
                           path_1=r'.\data_train\dcm\list1.csv',
                           path_m=r'.\data_ori\nii',
                           ki=ki, K=K, num_frame=num_frame, image_size=imsize, patch_size_tic=64, rawh=rawh, raww=raww, typ='val', transform=True, rand=False)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        if resume:
            v = torch.load(model_path)
        else:
            # v = TIC_LUV(img_size=imsize, patch_size=16, patch_size_tic=64, num_classes=2, num_frame=int(num_frame/2),
            #             attention_type='divided_space_time',
            #             pretrained_model='.\pretrained\jx_vit_base_p16_224-80ecf9dd.pth')
            v = TIC_LUV(img_size=imsize, patch_size=16, embed_dim=768, num_classes=2, num_frame=num_frame,
                        attention_type='divided_space_time',
                        pretrained_model='.\pretrained\jx_vit_base_p16_224-80ecf9dd.pth')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        v.to(device)
        CELoss = FocalLoss()
        MDLoss = MMDLoss()
        # CELoss = nn.CrossEntropyLoss()
        optimizer = optim.SGD([{'params': v.parameters(), 'lr': 0.002}], momentum=0.9, weight_decay=5e-4)
        # optimizer = optim.AdamW(v.parameters(), lr=0.002, betas=(0.5, 0.9) )
        max_val_acc = 0
        lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]

        for epoch in range(start_epoch, nb_epoch):
            print('\nEpoch: %d' % epoch)
            v.train()
            train_loss = 0
            correct = 0
            total = 0
            idx = 0
            for batch_idx, (inputs, tics, targets, patient, cp) in enumerate(trainloader):
                tics = tics.view(1, 6, 16).float()
                inputs = inputs.permute(0, 2, 1, 3, 4).float()  # BTCHW->BCTHW
                inputs1 = inputs[:, :, :, :, imsize:]  # CEUS
                inputs2 = inputs[:, :, :, :, :imsize, ]  # GSUS
                idx = batch_idx
                cp = torch.from_numpy(np.array(cp))
                if inputs2.shape[0] < batch_size:
                    continue
                if use_cuda:
                    inputs1, inputs2, tics, targets, cp = inputs1.to(device), inputs2.to(device), tics.to(device), targets.to(device), cp.to(device)
                inputs1, inputs2, tics, targets, cp = Variable(inputs1), Variable(inputs2), Variable(tics), Variable(targets), Variable(cp)

                for nlr in range(len(optimizer.param_groups)):
                    optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])
                optimizer.zero_grad()
                output_concat, fea_vd, fea_tic = v(inputs1, inputs2, tics, cp)
                concat_loss = 10*CELoss(output_concat, targets) + 0.001*MDLoss(fea_vd, fea_tic)
                concat_loss.backward()
                optimizer.step()

                _, predicted = torch.max(output_concat.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()
                train_loss += concat_loss.item()

                if batch_idx % 5 == 0:
                    print('K-fold %d, Epoch %d, Step: %d| Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        ki, epoch, batch_idx, train_loss / (batch_idx + 1),
                        100. * float(correct) / total, correct, total))

            train_acc = 100. * float(correct) / total
            train_loss = train_loss / (idx + 1)

            writer.add_scalar('train/train_loss', train_loss, epoch)
            writer.add_scalar('train/train_acc', train_acc, epoch)
            with open(exp_dir + '/results_train_np_%d.txt' % ki, 'a') as file:
                file.write('K-fold %d, Epoch %d | train_acc = %.5f | train_loss = %.5f\n' % (
                    ki, epoch, train_acc, train_loss))
            torch.cuda.empty_cache()


            # val ------------------------------------------
            testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)
            v.eval()
            test_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, tics, targets, patient, cp) in enumerate(testloader):
                tics = tics.view(1, 6, 16).float()
                inputs = inputs.permute(0, 2, 1, 3, 4).float()  # BTCHW->BCTHW
                inputs1 = inputs[:, :, :, :, imsize:]  # CEUS
                inputs2 = inputs[:, :, :, :, :imsize, ]  # GSUS

                if use_cuda:
                    inputs1, inputs2, tics, targets, cp = inputs1.to(device), inputs2.to(device), tics.to(device), targets.to(device), cp.to(device)
                inputs1, inputs2, tics, targets, cp = Variable(inputs1), Variable(inputs2), Variable(tics), Variable(targets), Variable(cp)
                output, out_vd, out_tic = v(inputs1, inputs2, tics, cp)
                loss = CELoss(output, targets) + MDLoss(out_vd, out_tic)
        
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()
                # print('patient:', patient.cpu(), 'label:', targets.data.cpu(), 'pred:', predicted.cpu())
        
                if batch_idx % 20 == 0:
                    print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
        
            test_acc = 100. * float(correct) / total
            test_loss = test_loss / (batch_idx + 1)

            writer.add_scalar('test/test_loss', test_loss, epoch)
            writer.add_scalar('test/test_acc', test_acc, epoch)
            with open(exp_dir + '/results_test_np_%d.txt' % ki, 'a') as file:
                file.write('K-fold %d, Epoch %d | test_acc = %.5f | test_loss = %.5f\n' % (
                    ki, epoch, test_acc, test_loss))

            if test_acc > max_val_acc:
                print('Best test accuracy, %f' % test_acc)
                max_val_acc = test_acc
                v.cpu()
                torch.save(v, './' + store_name + '/model_cp_2modal_focal_fold_%d.pth'%ki)
                v.to(device)

        torch.cuda.empty_cache()

train(nb_epoch=70,             # number of epoch
      batch_size=1,            # batch size
      store_name='bird',       # folder for output
      resume=False,            # resume training from checkpoint
      start_epoch=0,           # the start epoch number when you resume the training
      model_path=r'.\bird\model_cp_2modal_focal_fold0.pth',
      imsize=256,
      num_frame=8, #16,
      rawh=896,
      raww=704)         # center crop (896, 2*704) from raw dcm, when patch size of TICA is 64
