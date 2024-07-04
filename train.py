import math
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from oid import OID
import torchvision.transforms.v2 as transforms

estimate_fig = None


def callback(**kwargs):
    fig.clear()
    fig.text(0.02, 0.02, f"Epoch: {kwargs['epoch']}, step: {kwargs['step']}, {kwargs['type']} loss: {kwargs.get('loss', None)}")
    ax = fig.add_subplot(spec[0])
    ax.imshow(np.array(oid.B.detach().cpu()).transpose(1, 2, 0), vmin=0, vmax=1, cmap='gray')
    ax = fig.add_subplot(spec[1])
    ax.imshow(np.array(oid.I.detach().cpu()).transpose(1, 2, 0), vmin=0, vmax=1, cmap='gray')
    ax = fig.add_subplot(spec[2])
    ax.imshow(np.array(oid.W.detach().cpu()).transpose(1, 2, 0), vmin=0, vmax=1, cmap='gray')
    ax = fig.add_subplot(spec[3])
    ax.imshow(np.array(oid.K.detach().cpu()).transpose(1, 2, 0), cmap='gray')
    plt.pause(0.01)


def estimate_callback(latent_image, epoch):
    global estimate_fig
    if estimate_fig:
        estimate_fig.clear()
    else:
        estimate_fig = plt.figure()
    estimate_fig.text(0.02, 0.02, f"Epoch: {epoch}")
    ax = estimate_fig.add_subplot()
    ax.imshow(np.array(latent_image))
    plt.pause(0.01)

if __name__ == '__main__':    
    plt.ion()
    fig = plt.figure(figsize=(11, 3))
    spec = fig.add_gridspec(1, 4, width_ratios=[3, 3, 3, 1.5])
    fig.show()

    # kernels = np.ceil(kernel_size ** np.array(1 / np.sqrt(2) ** np.arange(max_iter))).astype(int)
    image = Image.open('./example.png')
    blurred_image = (np.asarray(image) / 255).transpose(2, 0, 1)
    height, width = image.size

    kernels_size = [7, 11, 15, 21, 27] 
    
    kernel = np.loadtxt('./kernel.csv', delimiter=',')
    # latent_image = np.loadtxt('./image.csv', delimiter=',')#[::-1, ::-1]
    latent_image = np.array(Image.open('./latent.bmp')).transpose(2, 0, 1) / 255

    # oid = OID(blurred_image.mean(0, keepdims=True), kernels_size, device='cuda')
    # oid.I.data = torch.nn.functional.interpolate(torch.tensor(latent_image.copy(), dtype=torch.float32, device='cuda').expand(1, 3, -1, -1), size=oid.image_size, mode='bilinear').squeeze(0).mean(0, keepdims=True)
    oid = OID(blurred_image, kernels_size, device='cuda', latent_type='gray')
    # oid.I.data = torch.nn.functional.interpolate(torch.tensor(latent_image.copy().mean(axis=0, keepdims=True), dtype=torch.float32, device='cuda').unsqueeze(0), size=oid.image_size, mode='bilinear').squeeze(0)
    # oid.K.data = torch.tensor(kernel, dtype=torch.float32, device='cuda').unsqueeze(0)
    oid.train(lr=1e-6, callback=callback)

    image = Image.fromarray((oid.estimate_latent(callback=estimate_callback) * 255).astype(np.uint8))
    image.save('recover.bmp')
    
    plt.ioff()
    plt.show()
