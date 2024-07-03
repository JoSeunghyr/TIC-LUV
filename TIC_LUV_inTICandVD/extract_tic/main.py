import torch
from datasets import KZDataset
def train(batch_size, imsize=256, num_frame=16, rawh=896, raww=704):

    for ki in range(1):

        print('==> Preparing data..')

        trainset = KZDataset(path_0=r'list_0.csv',
                             path_1=r'list_1.csv',
                             path_m=r'.\nii',
                             num_frame=num_frame, image_size=imsize, patch_size_tic=64, rawh=rawh, raww=raww, typ='train', transform=True, rand=False)
        valset = KZDataset(path_0=r'list_0.csv',
                           path_1=r'list_1.csv',
                           path_m=r'.\nii',
                           num_frame=num_frame, image_size=imsize, patch_size_tic=64, rawh=rawh, raww=raww, typ='val', transform=True, rand=False)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        for epoch in range(1):
            print('\nEpoch: %d' % epoch)

            for batch_idx, (inputs, tics, targets, patient, cp) in enumerate(trainloader):
                print(patient[0])
                tics = tics.view(1, 6, 16).float()
                torch.save(tics, './small_tics/' + patient[0] + '.pt')

            # val ------------------------------------------
            testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)

            for batch_idx, (inputs, tics, targets, patient, cp) in enumerate(testloader):
                print(patient[0])
                tics = tics.view(1, 6, 16).float()
                torch.save(tics, './small_tics/' + patient[0] + '.pt')
        torch.cuda.empty_cache()

train(batch_size=1,            # batch size
      imsize=256,
      num_frame=8, #16,
      rawh=896,
      raww=704)         # center crop (896, 2*704) from raw dcm, when patch size of TICA is 64
