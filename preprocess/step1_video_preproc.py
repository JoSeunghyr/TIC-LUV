import os
from PIL import Image
import SimpleITK as sitk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageStat
import scipy.signal as sg
import math
import csv

def file_list(dirname, ext='.csv'):
    return list(filter(
        lambda filename: os.path.splitext(filename)[1] == ext,
        os.listdir(dirname)))


def clubrt(img):
    bimg = Image.fromarray(img)
    stat = ImageStat.Stat(bimg)
    r, g, b = stat.rms
    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))


def imgtic(images):
    """
    calculate tics of CEUS videos
    """
    frame = len(images)
    tic_list = []
    for i in range(frame):
        fimg = images[i]
        brt = clubrt(fimg)  # calculate tic of each frame
        brt = round(brt, 2)
        tic_list.append(brt)
    tic_list = np.array(tic_list)
    return tic_list

def savimgtic(path, brt_list, patient):
    """
    save tics as .png
    """
    x_axis_data = np.arange(0, brt_list.shape[0])
    y_axis_data = brt_list
    fig = plt.figure()
    plt.plot(x_axis_data, y_axis_data, '-', color='orange', alpha=0.8, linewidth=1.5)
    plt.xlabel('Frame', fontsize='large')
    plt.ylabel('Brightness', fontsize='large')
    plt.xlim(0, brt_list.shape[0] + 1)
    plt.ylim(0, max(brt_list) + 1)
    #plt.show()
    # os.makedirs(path + '/nflt/' , exist_ok=True)
    fig.savefig(path + '/nflt/' + patient + '.png')
    plt.close()

# def phticflt(tic, windowsize):
#     window = np.ones(int(windowsize)) / float(windowsize)
#     ntic = np.convolve(tic, window, 'valid')
#     return ntic
'''Savitzky-Golay Filter'''
def sgticflt(tic, windowsize):
    ntic = sg.savgol_filter(tic, windowsize, 3, mode='nearest')
    return ntic


def savflttic(path, brt_list, patient):
    """
    save smooth tics
    """
    x_axis_data = np.arange(0, brt_list.shape[0])
    y_axis_data = brt_list
    fig = plt.figure()
    plt.plot(x_axis_data, y_axis_data, '-', color='orange', alpha=0.8, linewidth=1.5)
    plt.xlabel('Frame', fontsize='large')
    plt.ylabel('Brightness', fontsize='large')
    plt.xlim(0, brt_list.shape[0] + 1)
    plt.ylim(0, max(brt_list) + 1)
    #plt.show()
    #os.makedirs(path + '/flt/' , exist_ok=True)
    fig.savefig(path + '/flt/' + patient + '.png')
    plt.close()


# click tts, ttp
def onclick(event):
    if event.button == 1:  # left click
        ix, iy = event.xdata, event.ydata
        clicked_points.append((ix, iy))
        print(f"Clicked at x={ix}, y={iy}")
        # draw red circle on click point
        ax.plot(ix, iy, 'ro')
        fig.canvas.draw()
        if len(clicked_points) == 2:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

def resizekeysample(path, rawdcm, ps, pp, len, size):
    tts = int(ps[0])
    ttp = int(pp[0])
    h = size
    w = 2 * size
    kong = np.zeros((len, h, w, 3), np.uint8)
    quo, rem = divmod(ttp - tts, len)
    if rem <= 10:
        end = ttp - rem
    else:
        end = ttp + (len - rem)
    newquo = int((end - tts) / len)
    for ii in range(len):
        img_resize = cv2.resize(rawdcm[tts + ii * newquo, :, :, :], (w, h))
        kong[ii, :, :, :] = img_resize
    # print(kong.shape)
    kong = sitk.GetImageFromArray(kong)
    sitk.WriteImage(kong, path + '/' + patient +'.dcm')


def keysample(path, rawdcm, ps, pp, len):
    tts = int(ps[0])
    ttp = int(pp[0])
    h = rawdcm.shape[1]
    w = rawdcm.shape[2]
    kong = np.zeros((len, h, w, 3), np.uint8)
    quo, rem = divmod(ttp - tts, len)
    if rem <= 10:
        end = ttp - rem
    else:
        end = ttp + (len - rem)
    newquo = int((end - tts) / len)
    for ii in range(len):
        img_resize = rawdcm[tts + ii * newquo, :, :, :]
        kong[ii, :, :, :] = img_resize
    # print(kong.shape)
    kong = sitk.GetImageFromArray(kong)
    sitk.WriteImage(kong, path + '/' + patient + '.dcm')

root_path = '../data_ori/dcm'  # original dcm data
imtic_path = '../data_ori/img_tics'  # save tics of CEUS videos
dcm_path = '../data_train/dcm'  # save processed videos

for i in file_list(root_path, '.dcm'):
    patient = i.split('.')[0]
    img_path = root_path+'/'+i
    img1 = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img1)  # t,h,w,c
    frames = img.shape[0]
    dw = int(img.shape[2]/2)  # 721
    hui = img[:, :, :dw, :]
    # hui_img = hui[50]
    # plt.imshow(hui_img)
    zy = img[:,:,dw:,:]
    dcmtic = imgtic(zy)
    savimgtic(imtic_path, dcmtic, patient)  # save raw tics
    # edcmtic = phticflt(dcmtic, 25)
    '''Savitzky-Golay'''
    edcmtic = sgticflt(dcmtic, 31)
    savflttic(imtic_path, edcmtic, patient)  # save smooth tics
    '''tts,ttp'''
    fig, ax = plt.subplots()
    ax.plot(edcmtic, label='Intensity')
    ax.set_xlabel('Time')
    ax.set_ylabel('Intensity')
    ax.set_title('Time-Intensity Curve')
    ax.legend()
    clicked_points = []
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    os.makedirs(imtic_path + '/ttsttp', exist_ok=True)
    fig.savefig(imtic_path + '/ttsttp/' + patient + '.png')
    with open(imtic_path + '/ttsttp.csv', mode='a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([patient, clicked_points[0], clicked_points[1]])
    print("Saved tts ttp of %s" % patient)
    '''sample rawdcm'''
    # dcm = resizekeysample(dcm_path, img, clicked_points[0], clicked_points[1], 16, 256)
    dcm = keysample(dcm_path, img, clicked_points[0], clicked_points[1], 16)
    continue

print("FINISHED")