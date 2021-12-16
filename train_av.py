import argparse
from pathlib import Path
import time
import wandb
import torch
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from models_av import MTCN_AV
from epic_kitchens import EpicKitchens
from egtea import Egtea
from mixup import mixup_data, mixup_criterion
from utils import accuracy, multitask_accuracy, save_checkpoint, AverageMeter


_DATASETS = {'epic': EpicKitchens, 'egtea': Egtea}
_NUM_CLASSES = {'epic-55': [125, 352], 'epic-100': [97, 300], 'egtea': 106}

parser = argparse.ArgumentParser(description=('Train Audio-Visual Transformer on Sequence ' +
                                              'of actions from untrimmed video'))

# ------------------------------ Dataset -------------------------------
parser.add_argument('--train_hdf5_path', type=Path)
parser.add_argument('--val_hdf5_path', type=Path)
parser.add_argument('--train_pickle', type=Path)
parser.add_argument('--val_pickle', type=Path)
parser.add_argument('--dataset', choices=['epic-55', 'epic-100', 'egtea'])
# ------------------------------ Model ---------------------------------
parser.add_argument('--seq_len', type=int, default=5)
parser.add_argument('--visual_input_dim', type=int, default=2304)
parser.add_argument('--audio_input_dim', type=int, default=2304)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--dim_feedforward', type=int, default=2048)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--classification_mode', choices=['summary', 'all'], default='summary')
parser.add_argument('--dropout', type=float, default=0.1)
# ------------------------------ Train ----------------------------------
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# ------------------------------ Optimizer ------------------------------
parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[25, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
# ------------------------------ Misc ------------------------------------
parser.add_argument('--output_dir', type=Path)
parser.add_argument('--disable_wandb_log', action='store_true')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')

args = parser.parse_args()

best_prec1 = 0
training_iterations = 0

if not args.output_dir.exists():
    args.output_dir.mkdir(parents=True)


def main():
    global args, best_prec1

    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
    model = MTCN_AV(_NUM_CLASSES[args.dataset],
                    seq_len=args.seq_len,
                    num_clips=1,
                    visual_input_dim=args.visual_input_dim,
                    audio_input_dim=args.audio_input_dim if args.dataset.split('-')[0] == 'epic' else None,
                    d_model=args.d_model,
                    dim_feedforward=args.dim_feedforward,
                    nhead=args.nhead,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    classification_mode=args.classification_mode,
                    audio=not args.dataset == 'egtea')
    model = model.to(device)

    if not args.disable_wandb_log:
        wandb.init(project='MTCN', config=args)
        wandb.watch(model)

    dataset = _DATASETS[args.dataset.split('-')[0]]
    train_loader = torch.utils.data.DataLoader(
        dataset(args.train_hdf5_path,
                args.train_pickle,
                visual_feature_dim=args.visual_input_dim,
                audio_feature_dim=args.audio_input_dim if args.dataset.split('-')[0] == 'epic' else None,
                window_len=args.seq_len,
                num_clips=10,
                clips_mode='random',
                labels_mode='all' if args.classification_mode == 'all' else 'center_action',),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataset(args.val_hdf5_path,
                args.val_pickle,
                visual_feature_dim=args.visual_input_dim,
                audio_feature_dim=args.audio_input_dim if args.dataset.split('-')[0] == 'epic' else None,
                window_len=args.seq_len,
                num_clips=10,
                clips_mode='random',
                labels_mode='all' if args.classification_mode == 'all' else 'center_action',),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not args.classification_mode == 'all':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, args.lr_steps, gamma=0.1)

    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, device)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, device)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, args.output_dir)
        scheduler.step()


def train(train_loader, model, criterion, optimizer, epoch, device):
    global training_iterations
    is_multitask = isinstance(model.num_class, list)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if is_multitask:
        verb_losses = AverageMeter()
        noun_losses = AverageMeter()
    if args.classification_mode == 'all':
        if 'epic' in args.dataset:
            weights = torch.tensor(2 * args.seq_len * [0.1] + [0.9]).unsqueeze(0).cuda(device=0)
        else:
            weights = torch.tensor(args.seq_len * [0.1] + [0.9]).unsqueeze(0).cuda(device=0)
    else:
        weights = None

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        input, target_a, target_b, lam = mixup_data(input, target, alpha=0.2)
        # compute output
        output = model(input)
        batch_size = input.size(0)
        if not is_multitask:
            target_a = target_a.to(device)
            target_b = target_b.to(device)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam, weights=weights)
        else:
            target_a = {k: v.to(device) for k, v in target_a.items()}
            target_b = {k: v.to(device) for k, v in target_b.items()}
            loss_verb = mixup_criterion(criterion, output[0], target_a['verb'], target_b['verb'], lam, weights=weights)
            loss_noun = mixup_criterion(criterion, output[1], target_a['noun'], target_b['noun'], lam, weights=weights)
            loss = 0.5 * (loss_verb + loss_noun)
            verb_losses.update(loss_verb.item(), batch_size)
            noun_losses.update(loss_noun.item(), batch_size)
        losses.update(loss.item(), batch_size)
        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        training_iterations += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if not is_multitask:
                if not args.disable_wandb_log:
                    wandb.log(
                        {
                            "Train/loss": losses.avg,
                            "Train/epochs": epoch,
                            "Train/lr": optimizer.param_groups[-1]['lr'],
                            "train_step": training_iterations,
                        },
                    )

                message = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' +
                           'Time {batch_time.avg:.3f} ({batch_time.avg:.3f})\t' +
                           'Data {data_time.avg:.3f} ({data_time.avg:.3f})\t' +
                           'Loss {loss.avg:.4f} ({loss.avg:.4f})\t'
                           ).format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses,
                        lr=optimizer.param_groups[-1]['lr'])
            else:
                if not args.disable_wandb_log:
                    wandb.log(
                        {
                            "Train/loss": losses.avg,
                            "Train/epochs": epoch,
                            "Train/lr": optimizer.param_groups[-1]['lr'],
                            "Train/verb/loss": verb_losses.avg,
                            "Train/noun/loss": noun_losses.avg,
                            "train_step": training_iterations,
                        },
                    )
                message = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' +
                           'Time {batch_time.avg:.3f} ({batch_time.avg:.3f})\t' +
                           'Data {data_time.avg:.3f} ({data_time.avg:.3f})\t' +
                           'Loss {loss.avg:.4f} ({loss.avg:.4f})\t' +
                           'Verb Loss {verb_loss.avg:.4f} ({verb_loss.avg:.4f})\t' +
                           'Noun Loss {noun_loss.avg:.4f} ({noun_loss.avg:.4f})\t'  # +
                           ).format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, verb_loss=verb_losses,
                    noun_loss=noun_losses,
                    lr=optimizer.param_groups[-1]['lr'])

            print(message)


def validate(val_loader, model, criterion, device, name=''):
    global training_iterations
    is_multitask = isinstance(model.num_class, list)
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        if is_multitask:
            verb_losses = AverageMeter()
            noun_losses = AverageMeter()
            verb_top1 = AverageMeter()
            verb_top5 = AverageMeter()
            noun_top1 = AverageMeter()
            noun_top5 = AverageMeter()
        if args.classification_mode == 'all':
            if 'epic' in args.dataset:
                weights = torch.tensor(2 * args.seq_len * [0.1] + [0.9]).unsqueeze(0).cuda(device=0)
            else:
                weights = torch.tensor(args.seq_len * [0.1] + [0.9]).unsqueeze(0).cuda(device=0)
        else:
            weights = None
        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target, _) in enumerate(val_loader):

            input = input.to(device)

            # compute output
            output = model(input)
            batch_size = input.size(0)
            if not is_multitask:
                target = target.to(device)
                loss = criterion(output, target)
                if weights is not None:
                    loss = loss * weights
                    loss = loss.sum(1)
                    loss = loss.mean()
                output = output if len(output.shape) == 2 else output[:, :, -1]
                target = target if len(target.shape) == 1 else target[:, -1]
                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
            else:
                target = {k: v.to(device) for k, v in target.items()}
                loss_verb = criterion(output[0], target['verb'])
                if weights is not None:
                    loss_verb = loss_verb * weights
                    loss_verb = loss_verb.sum(1)
                    loss_verb = loss_verb.mean()
                loss_noun = criterion(output[1], target['noun'])
                if weights is not None:
                    loss_noun = loss_noun * weights
                    loss_noun = loss_noun.sum(1)
                    loss_noun = loss_noun.mean()
                loss = 0.5 * (loss_verb + loss_noun)
                verb_losses.update(loss_verb.item(), batch_size)
                noun_losses.update(loss_noun.item(), batch_size)

                verb_output = output[0] if len(output[0].shape) == 2 else output[0][:, :, -1]
                noun_output = output[1] if len(output[1].shape) == 2 else output[1][:, :, -1]
                verb_target = target['verb'] if len(target['verb'].shape) == 1 else target['verb'][:, -1]
                noun_target = target['noun'] if len(target['noun'].shape) == 1 else target['noun'][:, -1]
                verb_prec1, verb_prec5 = accuracy(verb_output, verb_target, topk=(1, 5))
                verb_top1.update(verb_prec1, batch_size)
                verb_top5.update(verb_prec5, batch_size)

                noun_prec1, noun_prec5 = accuracy(noun_output, noun_target, topk=(1, 5))
                noun_top1.update(noun_prec1, batch_size)
                noun_top5.update(noun_prec5, batch_size)

                prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                                  (verb_target, noun_target),
                                                  topk=(1, 5))

            losses.update(loss.item(), batch_size)
            top1.update(prec1, batch_size)
            top5.update(prec5, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if not is_multitask:
            if not args.disable_wandb_log:
                wandb.log(
                    {
                        "Val/loss": losses.avg,
                        "Val/Top1_acc": top1.avg,
                        "Val/Top5_acc": top5.avg,
                        "val_step": training_iterations,
                    },
                )

            message = ('Testing Results: '
                       'Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} '
                       'Loss {loss.avg:.5f}').format(top1=top1,
                                                     top5=top5,
                                                     loss=losses)
        else:
            if not args.disable_wandb_log:
                wandb.log(
                    {
                        "Val/loss": losses.avg,
                        "Val/Top1_acc": top1.avg,
                        "Val/Top5_acc": top5.avg,
                        "Val/verb/loss": verb_losses.avg,
                        "Val/verb/Top1_acc": verb_top1.avg,
                        "Val/verb/Top5_acc": verb_top5.avg,
                        "Val/noun/loss": noun_losses.avg,
                        "Val/noun/Top1_acc": noun_top1.avg,
                        "Val/noun/Top5_acc": noun_top5.avg,
                        "val_step": training_iterations,
                    },
                )

            message = ("Testing Results: "
                       "{name} Verb Prec@1 {verb_top1.avg:.3f} Verb Prec@5 {verb_top5.avg:.3f} "
                       "{name} Noun Prec@1 {noun_top1.avg:.3f} Noun Prec@5 {noun_top5.avg:.3f} "
                       "{name} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} "
                       "{name} Verb Loss {verb_loss.avg:.5f} "
                       "{name} Noun Loss {noun_loss.avg:.5f} "
                       "{name} Loss {loss.avg:.5f}").format(verb_top1=verb_top1, verb_top5=verb_top5,
                                                            noun_top1=noun_top1, noun_top5=noun_top5,
                                                            top1=top1, top5=top5,
                                                            name=name,
                                                            verb_loss=verb_losses,
                                                            noun_loss=noun_losses,
                                                            loss=losses)
        print(message)
        

        return top1.avg


if __name__ == '__main__':
    main()
