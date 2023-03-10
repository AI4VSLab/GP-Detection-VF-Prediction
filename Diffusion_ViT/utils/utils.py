import os
import numpy as np
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
from torchvision import utils as tv_utils
import torch
import torchvision as tv
from .eval_quality import eval_is_fid
from tqdm import tqdm
from ExpUtils import wlog


def sqrt(x):
    return int(torch.sqrt(torch.Tensor([x])))


def plot(p, x):
    return tv.utils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))


def cycle(loader):
    while True:
        for data in loader:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def save_sample_q(model, epoch, arg, num=100, save=True, i=0, video=False):
    milestone = str(epoch)
    if i > 0:
        milestone += '-' + str(i)
    batches = num_to_groups(num, num)
    all_images_list = list(map(lambda n: model.sample(batch_size=n, save_video=video), batches))
    all_images = torch.cat(all_images_list, dim=0)
    if save:
        tv_utils.save_image(all_images, os.path.join(arg.save_path, 'samples', f'sample-{milestone}.png'), nrow=int(sqrt(num)))
    return all_images


def to_video(ema_model, arg):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.animation as animation
    import matplotlib.image as mpimg

    ema_model.eval()
    num = 100
    if arg.dataset != 'cifar10':
        num = 25
    save_sample_q(ema_model, 0, arg, num=num, video=True)

    frames = []
    fig = plt.figure()
    for i in range(200):
        img = mpimg.imread('iter-%d.png' % (i + 1))
        img = plt.imshow(img, animated=True)
        frames.append([img])

    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True, repeat_delay=0)
    ani.save('%s-generate.mp4' % arg.dataset)
    print("save a video to show the sampling")
    for i in range(200):
        os.remove('iter-%d.png' % (i + 1))


def sample_ema(ema_model, buffer, epoch, arg, title='ema'):
    ema_model.eval()
    q = save_sample_q(ema_model, epoch, arg)
    idx_start = 0
    if epoch > 20:
        idx_start = (epoch - 20) % 100 * (100 if arg.dataset != 'stl10' else 50)
    buffer[idx_start:idx_start + 100] = q
    inc_score, fid = 0, 0
    if title:
        torch.save({'model_state_dict': ema_model.state_dict()}, os.path.join(arg.save_path, 'ema_checkpoint.pth'))
        if (epoch * 10) % arg.epochs == 0 or epoch <= 10:
            torch.save({'model_state_dict': ema_model.state_dict()}, os.path.join(arg.save_path, f'ema_{epoch}_checkpoint.pth'))
    if epoch and epoch % 5 == 0 and not arg.no_fid:  # and epoch >= 50
        end = (idx_start + 100) if epoch < 120 else 10000
        if arg.dataset == 'stl10':
            end = (idx_start + 100) if epoch < 70 else 5000
        metrics = eval_is_fid((buffer[:end] + 1) * 127.5, dataset=arg.dataset)
        inc_score = metrics['inception_score_mean']
        fid = metrics['frechet_inception_distance']
        if title is None:
            arg.writer.add_scalar('GEN/model_IS', inc_score, epoch)
            arg.writer.add_scalar('GEN/model_FID', fid, epoch)
        else:
            arg.writer.add_scalar('GEN/IS', inc_score, epoch)
            arg.writer.add_scalar('GEN/FID', fid, epoch)
        wlog('Epoch %d  IS, FID: %.3f, %.3f' % (epoch, inc_score, fid))
    return inc_score, fid


# def sample_ema_tf(ema_model, buffer, epoch, arg, num=10):
#     from Task.eval_buffer import eval_is_fid as eval_is_fid_tf
#     ema_model.eval()
#     q = save_sample_q(ema_model, epoch, arg, num=10)
#     # idx_start = 0
#     # if epoch > 20:
#     #     idx_start = (epoch - 20) % 100 * 100
#     # buffer[idx_start:idx_start + 100] = q
#     inc_score, fid = 0, 0
#     torch.save({'model_state_dict': ema_model.state_dict()}, os.path.join(arg.save_path, 'ema_checkpoint.pth'))
#
#     if (epoch * 10) % arg.epochs == 0:
#         torch.save({'model_state_dict': ema_model.state_dict()}, os.path.join(arg.save_path, f'ema_{epoch}_checkpoint.pth'))
#
#     # if epoch % 5 == 0 and epoch >= 100:
#     #     end = (idx_start + 100) if epoch < 120 else 10000
#     #     _, _, fid = eval_is_fid_tf(buffer[:end], arg, eval='fid')
#     #     arg.writer.add_scalar('GEN/FID', fid, epoch)
#     #     wlog('Epoch %d  FID: %.3f' % (epoch, fid))
#     return fid
