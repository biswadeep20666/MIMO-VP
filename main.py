import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Models import get_model
from Moving_mnist_dataset.moving_mnist import MovingMNIST
from skimage.metrics import structural_similarity as ssim
import os
import cv2
import preprocess
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='Moving_mnist_dataset', help='folder for dataset')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--checkpoint_path', type=str, default='/data/AAAI/MIMO-VP/checkpoints/epoch-879/model.ckpt-879', help='folder for dataset')
parser.add_argument('--lr', type=float, default=0.0005, help='learning_rate')
parser.add_argument('--n_epochs', type=int, default=1500, help='nb of epochs')
parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--eval_every', type=int, default=10, help='')
parser.add_argument('--save_dir', type=str, default='checkpoints')
parser.add_argument('--gen_frm_dir', type=str, default='results_mnist')
parser.add_argument('--patch_size', type=int, default=2)

#
parser.add_argument('-d_model', type=int, default=128)
parser.add_argument('-n_encoder_layers', type=int, default=6)
parser.add_argument('-n_decoder_layers', type=int, default=6)
parser.add_argument('-heads', type=int, default=8)
parser.add_argument('-dropout', type=int, default=0)


parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)

args = parser.parse_args()
output_length = args.total_length - args.input_length

if output_length <= 0:
    raise ValueError("total_length must be greater than input_length to produce forecast frames")

mm = MovingMNIST(
    root=args.root,
    is_train=True,
    n_frames_input=args.input_length,
    n_frames_output=output_length,
    num_objects=[2],
)
train_loader = torch.utils.data.DataLoader(
    dataset=mm, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4
)

mm = MovingMNIST(
    root=args.root,
    is_train=False,
    n_frames_input=args.input_length,
    n_frames_output=output_length,
    num_objects=[2],
)
test_loader = torch.utils.data.DataLoader(
    dataset=mm, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4
)


def split_decoder_layers(output, patch_channels):
    if output.size(2) % patch_channels != 0:
        raise ValueError(
            f"Decoder channels ({output.size(2)}) must be divisible by patch channels ({patch_channels})"
        )
    layers = output.size(2) // patch_channels
    return output.view(output.size(0), output.size(1), layers, patch_channels, *output.shape[3:])


def rollout_predictions(model, input_tensor, target_length):
    outputs = []
    produced = 0
    patch_channels = args.patch_size * args.patch_size
    current_input = input_tensor

    while produced < target_length:
        chunk_output = model(current_input)
        chunk_frames = min(chunk_output.size(1), target_length - produced)
        outputs.append(chunk_output[:, :chunk_frames])
        produced += chunk_frames

        layered_output = split_decoder_layers(chunk_output, patch_channels)
        current_input = layered_output[:, :, -1, :, :, :]

    return torch.cat(outputs, dim=1)


def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion1, criterion2):
    encoder_optimizer.zero_grad()
    output_image = rollout_predictions(encoder, input_tensor, target_tensor.size(1))
    expanded_target = torch.cat([target_tensor] * args.n_decoder_layers, dim=2)
    loss = 10 * criterion1(output_image, expanded_target) + criterion2(output_image, expanded_target)
    loss.backward()
    encoder_optimizer.step()
    return loss.item() / expanded_target.size(1)


def trainIters(encoder, n_epochs, print_every, eval_every):
    train_losses = []

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    criterion1 = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    criterion2 = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    itr = 0
    for epoch in range(0, n_epochs):
        t0 = time.time()
        loss_epoch = 0
        for i, out in enumerate(train_loader, 0):
            itr += 1
            input_tensor = out[1].to(device)
            input_tensor = preprocess.reshape_patch(input_tensor, args.patch_size)
            target_tensor = out[2].to(device)
            target_tensor = preprocess.reshape_patch(target_tensor, args.patch_size)
            loss = train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion1, criterion2)
            loss_epoch += loss

        train_losses.append(loss_epoch)
        if (epoch + 1) % print_every == 0:
            print('epoch ', epoch, ' loss ', loss_epoch, ' epoch time ', time.time() - t0)

        if (epoch + 1) % eval_every == 0:
            mse, mae, ssim_value = evaluate(encoder, test_loader)
            scheduler_enc.step(mse)
            stats = {'net_param': encoder.state_dict()}
            save_dir = os.path.join(args.save_dir, 'epoch-' + str(epoch))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            checkpoint_path = os.path.join(save_dir, 'model.ckpt' + '-' + str(epoch))
            torch.save(stats, checkpoint_path)
    return train_losses


def evaluate(encoder, loader):
    total_mse, total_mae, total_ssim = 0, 0, 0
    encoder.eval()
    patch_channels = args.patch_size * args.patch_size

    with torch.no_grad():
        for id, out in enumerate(loader, 0):
            input_tensor = out[1].to(device)
            input_tensor = preprocess.reshape_patch(input_tensor, args.patch_size)
            target_tensor = out[2].to(device)
            target_patched = preprocess.reshape_patch(target_tensor, args.patch_size)
            predictions_full = rollout_predictions(encoder, input_tensor, target_patched.size(1))
            layered_output = split_decoder_layers(predictions_full, patch_channels)
            predictions = layered_output[:, :, -1, :, :, :]
            predictions = preprocess.reshape_patch_back(predictions, args.patch_size)
            predictions = predictions.cpu().numpy()

            input_tensor = preprocess.reshape_patch_back(input_tensor, args.patch_size)
            input_np = input_tensor.cpu().numpy()
            target_np = target_tensor.cpu().numpy()

            if id < 20:
                path = os.path.join(args.gen_frm_dir, str(id))
                if not os.path.exists(path):
                    os.makedirs(path)
                for i in range(args.input_length):
                    name = 'gt' + str(i + 1) + '.png'
                    file_name = os.path.join(path, name)
                    img_gt = np.uint8(input_np[0, i, :, :, :] * 255)
                    img_gt = np.transpose(img_gt, [1, 2, 0])
                    cv2.imwrite(file_name, img_gt)

                for i in range(target_np.shape[1]):
                    name = 'gt' + str(i + 1 + args.input_length) + '.png'
                    file_name = os.path.join(path, name)
                    img_gt = np.uint8(target_np[0, i, :, :, :] * 255)
                    img_gt = np.transpose(img_gt, [1, 2, 0])
                    cv2.imwrite(file_name, img_gt)

                for i in range(predictions.shape[1]):
                    name = 'pd' + str(i + 1 + args.input_length) + '.png'
                    file_name = os.path.join(path, name)
                    img_pd = predictions[0, i, :, :, :]
                    img_pd = np.maximum(img_pd, 0)
                    img_pd = np.minimum(img_pd, 1)
                    img_pd = np.uint8(img_pd * 255)
                    img_pd = np.transpose(img_pd, [1, 2, 0])
                    cv2.imwrite(file_name, img_pd)

            mse_batch = np.mean((predictions - target_np) ** 2, axis=(0, 1, 2)).sum()
            mae_batch = np.mean(np.abs(predictions - target_np), axis=(0, 1, 2)).sum()
            total_mse += mse_batch
            total_mae += mae_batch

            for a in range(0, target_np.shape[0]):
                for b in range(0, target_np.shape[1]):
                    total_ssim += ssim(target_np[a, b, 0,], predictions[a, b, 0,]) / (target_np.shape[0] * target_np.shape[1])

    print('eval mse ', total_mse / len(loader), ' eval mae ', total_mae / len(loader), ' eval ssim ', total_ssim / len(loader))
    return total_mse / len(loader), total_mae / len(loader), total_ssim / len(loader)


print('BEGIN TRAIN')

model = get_model(args).to(device)
model = nn.DataParallel(model).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


total_params = count_parameters(model)
print('encoder ', total_params)

if args.checkpoint_path != '':
    print('load model:', args.checkpoint_path)
    stats = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(stats['net_param'])
    mse, mae, ssim_value = evaluate(model, test_loader)
else:
    plot_losses = trainIters(model, args.n_epochs, print_every=args.print_every, eval_every=args.eval_every)
