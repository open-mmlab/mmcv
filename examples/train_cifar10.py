from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.torchpack import Runner
from torchvision import datasets, transforms

import resnet_cifar


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def batch_processor(model, data, train_mode):
    img, label = data
    label = label.cuda(non_blocking=True)
    pred = model(img)
    loss = F.cross_entropy(pred, label)
    acc_top1, acc_top5 = accuracy(pred, label, topk=(1, 5))
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    log_vars['acc_top1'] = acc_top1.item()
    log_vars['acc_top5'] = acc_top5.item()
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=img.size(0))
    return outputs


def parse_args():
    parser = ArgumentParser(description='Train CIFAR-10 classification')
    parser.add_argument('config', help='train config file path')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = getattr(resnet_cifar, cfg.model)()
    model = torch.nn.DataParallel(model, device_ids=cfg.gpus).cuda()

    normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
    train_dataset = datasets.CIFAR10(
        root=cfg.data_root,
        train=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.CIFAR10(
        root=cfg.data_root,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    num_workers = cfg.data_workers * len(cfg.gpus)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    runner = Runner(model, cfg.optimizer, batch_processor, cfg.work_dir)
    runner.register_default_hooks(
        lr_config=cfg.lr_policy,
        checkpoint_config=cfg.checkpoint_cfg,
        log_config=cfg.log_cfg)

    if cfg.get('resume_from') is not None:
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from') is not None:
        runner.load_checkpoint(cfg.load_from)

    runner.run([train_loader, val_loader], cfg.workflow, cfg.max_epoch)


if __name__ == '__main__':
    main()
