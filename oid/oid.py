import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import numpy as np


class OID:
    def __init__(self, blurred_image, kernels_size, device) -> None:
        self.kernels_size = kernels_size
        self.scales = [k / kernels_size[-1] for k in kernels_size]
        self.device = device

        self.set_image(blurred_image)

        self.weight_grad_I = 4e-4
        self.weight_K = 5.0
        # self.weight_K = 2.5e-3  # * 3
        self.weight_W = 1.8e-3
        self.weight_entropy_W = 2e-4

        self.epoch = 10
        self.epoch_I = 4
        self.epoch_K = 4

    def set_image(self, blurred_image):
        self.num_channels, *self.image_size = blurred_image.shape
        self.blurred_image = torch.tensor(blurred_image, dtype=torch.float32, device=self.device)

        # worse_image = torch.nn.functional.interpolate(self.blurred_image.unsqueeze(0), scale_factor=self.scales[0], mode='bilinear').squeeze(0)

        self.B = torch.nn.functional.interpolate(self.blurred_image.mean(dim=0, keepdim=True).unsqueeze(0),
                                                 size=self.image_size, mode='bilinear').squeeze(0)
        self.I = self.B.clone()
        self.W = torch.ones(1, self.I.shape[1], self.I.shape[2], device=self.device)
        self.K = torch.zeros(1, self.kernels_size[0], self.kernels_size[0], dtype=torch.float32, device=self.device)
        self.K[0, self.kernels_size[0] // 2, self.kernels_size[0] // 2 - 1:self.kernels_size[0] // 2 + 1] = 0.5
        # self.K = nn.Parameter(self.K)

    def compute_blurred_image(self, latent, kernel):
        computed_image = torch.conv2d(latent, kernel.expand(latent.shape[0], -1, -1, -1), padding=kernel.shape[-1] // 2, groups=latent.shape[0])
        # return torch.clamp(computed_image, 0.0, 1.0)
        return computed_image

    def compute_size(self, scale):
        return (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

    def optimize_W(self):
        self.W = torch.sigmoid(-((self.B.mean(0, keepdim=True) - self.compute_blurred_image(self.I, self.K).mean(0, keepdim=True))
                               ** 2 - self.weight_W) / self.weight_entropy_W).detach()

    def regularize_K(self):
        result = self.bwconncomp_2d(np.array(self.K.squeeze(0).detach().cpu() != 0.0, dtype=np.bool8))
        for r in result:
            if torch.sum(self.K[:, *r]) < 0.05: # 原文为 0.1
                self.K[:, *r] = 0.0
                pass
        self.K /= self.K.sum()
        # self.K.data /= self.K.sum()

    def grad_image(self, image):
        return image[:, :, :-1] - image[:, :, 1:], image[:, :-1, :] - image[:, 1:, :]

    # def configure_optimizers(self, lr):
    #     # return torch.optim.LBFGS([self.I], lr=2e-1), torch.optim.LBFGS([self.K], lr=1e-2)
    #     # return torch.optim.LBFGS([self.I], lr=1e-1), torch.optim.Adam([self.K], lr=1e-5, weight_decay=0.9)
    #     # return optim.LCG([self.I], eps=1e-5), optim.LCG([self.K], eps=1e-5)
    #     return torch.optim.LBFGS([self.I], lr=lr)

    def bwconncomp_2d(self, bw):
        PixelIdxList = []
        mask = np.zeros(bw.shape, dtype=np.bool8)

        while np.any(bw != mask):
            BW = bw != mask
            r0 = -1
            c0 = -1
            for r in range(BW.shape[0]):
                for c in range(BW.shape[1]):
                    if BW[r, c] == True:
                        r0, c0 = r, c
                        break
                if r0 != -1:
                    break

            idxlist = [(r0, c0)]

            mask[r0, c0] = True
            k = 0
            while k < len(idxlist):
                r, c = idxlist[k]
                if r - 1 >= 0:
                    if c - 1 >= 0 and BW[r - 1, c - 1] == True and (r - 1, c - 1) not in idxlist:
                        idxlist.append((r - 1, c - 1))
                        mask[r - 1, c - 1] = True
                    if c + 1 < BW.shape[1] and BW[r - 1, c + 1] == True and (r - 1, c + 1) not in idxlist:
                        idxlist.append((r - 1, c + 1))
                        mask[r - 1, c + 1] = True
                if r + 1 < BW.shape[0]:
                    if c - 1 >= 0 and BW[r + 1, c-1] == True and (r + 1, c - 1) not in idxlist:
                        idxlist.append((r + 1, c - 1))
                        mask[r + 1, c - 1] = True
                    if c + 1 < BW.shape[1] and BW[r + 1, c + 1] == True and (r + 1, c + 1) not in idxlist:
                        idxlist.append((r + 1, c + 1))
                        mask[r + 1, c + 1] = True
                k += 1
            a = np.array(idxlist, dtype=np.int64)
            PixelIdxList.append((a[:, 0].tolist(), a[:, 1].tolist()))
        return PixelIdxList

    def optimize(self, callback=None):
        for epoch in range(self.epoch):
            for step in range(self.epoch_I):
                # prev_I = self.I.clone()
                # prev_loss = 1e10

                def latent_loss(latent):
                    deblur_loss = torch.sum(self.W * (self.B - self.compute_blurred_image(latent, self.K)) ** 2)

                    grad_x_I, grad_y_I = self.grad_image(self.I.detach())
                    P_x = torch.max(torch.abs(grad_x_I), torch.tensor(1e-2)) ** -1.2
                    P_y = torch.max(torch.abs(grad_y_I), torch.tensor(1e-2)) ** -1.2  # eps=1e-5就不行 真离谱

                    grad_x_latent, grad_y_latent = self.grad_image(latent)
                    DI_loss = self.weight_grad_I * (torch.sum(P_x * grad_x_latent ** 2) +
                                                    torch.sum(P_y * grad_y_latent ** 2))
                    loss = deblur_loss + DI_loss
                    return loss

                grad_I = torch.func.jacrev(latent_loss)
                b = -grad_I(torch.zeros_like(self.I))
                self.I.data = self.conjugate_gradient(lambda x: grad_I(x) + b, b, self.I.clone(), 25)

                self.I.data.clamp_(0, 1)
                self.optimize_W()

                if callback is not None:
                    callback(**{
                        'epoch': epoch,
                        'step': step,
                        'type': 'I',
                        'loss': latent_loss(self.I).item()
                    })

            for step in range(self.epoch_K):
                self.optimize_W()

                grad_x_B, grad_y_B = self.grad_image(self.B)
                grad_x_I, grad_y_I = self.grad_image(self.I)
                # grad_I = torch.sqrt(grad_x_I[:, :-1, :] ** 2 + grad_y_I[:, :, :-1] ** 2)
                # grad_x_I[:, :-1, :][grad_I < 0.0188] = 0.0
                # grad_y_I[:, :, :-1][grad_I < 0.0188] = 0.0
                W_x = torch.sigmoid(-((grad_x_B.mean(0, keepdim=True) - self.compute_blurred_image(grad_x_I, self.K).mean(0, keepdim=True))
                                      ** 2 - self.weight_W) / self.weight_entropy_W)
                W_y = torch.sigmoid(-((grad_y_B.mean(0, keepdim=True) - self.compute_blurred_image(grad_y_I, self.K).mean(0, keepdim=True))
                                      ** 2 - self.weight_W) / self.weight_entropy_W)

                def kernel_loss(kerenl):
                    # self.K.data /= (self.K.sum() + 1e-5)
                    # deblur_loss = torch.sum(self.W * (self.B - self.compute_blurred_image(self.I, kerenl)) ** 2)
                    deblur_loss = torch.sum(W_x * (grad_x_B - self.compute_blurred_image(grad_x_I, kerenl)) ** 2) + \
                        torch.sum(W_y * (grad_y_B - self.compute_blurred_image(grad_y_I, kerenl)) ** 2)
                    K_loss = self.weight_K * torch.sum(kerenl ** 2)
                    return deblur_loss + K_loss

                grad_K = torch.func.jacrev(kernel_loss)
                b = -grad_K(torch.zeros_like(self.K))
                self.K = self.conjugate_gradient(lambda x: grad_K(x) + b, b, self.K, 21)

                self.K[self.K < 0.05 * self.K.max()] = 0.0
                self.K /= self.K.sum()

                if callback is not None:
                    callback(**{
                        'epoch': epoch,
                        'step': step,
                        'type': 'K',
                        'loss': kernel_loss(self.K).item()
                    })
            self.regularize_K()

    def conjugate_gradient(self, A, b, x, max_iter=15, tol=1e-4):
        r = b - A(x)
        p = r.clone()
        rs_old = torch.sum(r ** 2)

        for i in range(max_iter):
            Ap = A(p)
            alpha = rs_old / torch.sum(p * Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r ** 2)

            if torch.sqrt(rs_new) < tol:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x

    def train(self, lr, callback=None):
        for i, scale in enumerate(self.scales):
            if i > 0:
                self.I = torch.nn.functional.interpolate(self.I.unsqueeze(
                    0), size=self.compute_size(self.scales[i]), mode='bilinear').squeeze(0)
                self.W = torch.nn.functional.interpolate(self.W.unsqueeze(0), size=self.I.shape[-2:], mode='bilinear').squeeze(0)
                self.K = torch.nn.functional.interpolate(self.K.unsqueeze(0), size=(
                    self.kernels_size[i], self.kernels_size[i]), mode='bilinear').squeeze(0)
                self.regularize_K()
            self.optimize(callback)
            # W_loss = self.weight_W * torch.sum(torch.abs(1 - self.W))
            # W_entropy_loss = self.weight_entropy_W = torch.nn.functional.binary_cross_entropy_with_logits(self.W, self.W, reduction='sum')

    def estimate_latent(self, weight_grad_I=3e-3, callback=None):
        latent_image = self.blurred_image.clone()
        W = torch.sigmoid(-((self.blurred_image.mean(axis=0, keepdim=True) - self.compute_blurred_image(latent_image, self.K).mean(0, keepdim=True))
                            ** 2 - self.weight_W) / self.weight_entropy_W)
        for epoch in range(20):
            # prev_I = self.I.clone()
            # prev_loss = 1e10

            def latent_loss(latent):
                deblur_loss = torch.sum(W * (self.blurred_image - self.compute_blurred_image(latent, self.K)) ** 2)

                grad_x_I, grad_y_I = self.grad_image(latent_image)
                P_x = torch.max(torch.abs(grad_x_I), torch.tensor(1e-2)) ** -1.2
                P_y = torch.max(torch.abs(grad_y_I), torch.tensor(1e-2)) ** -1.2  # eps=1e-5就不行 真离谱

                grad_x_latent, grad_y_latent = self.grad_image(latent)
                DI_loss = weight_grad_I * (torch.sum(P_x * grad_x_latent ** 2) +
                                           torch.sum(P_y * grad_y_latent ** 2))
                loss = deblur_loss + DI_loss
                return loss

            grad_I = torch.func.jacrev(latent_loss)
            b = -grad_I(torch.zeros_like(latent_image))
            latent_image = self.conjugate_gradient(lambda x: grad_I(x) + b, b, latent_image, 25)

            latent_image.data.clamp_(0, 1)
            W = torch.sigmoid(-((self.blurred_image.mean(axis=0, keepdim=True) - self.compute_blurred_image(latent_image, self.K).mean(0, keepdim=True))
                                ** 2 - self.weight_W) / self.weight_entropy_W)

            if not callback is None:
                callback(np.array(latent_image.cpu()).transpose(1, 2, 0), epoch)

        return np.array(latent_image.cpu()).transpose(1, 2, 0)
