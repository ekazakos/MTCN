import argparse
import time
import wandb
import numpy as np
import pandas as pd
import random
from pathlib import Path

from corpus import EpicCorpus, EgteaCorpus
from models_lm import MTCN_LM
from utils import accuracy, multitask_accuracy, save_checkpoint, AverageMeter

import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

_NUM_CLASSES = {'epic-55': [125, 352], 'epic-100': [97, 300], 'egtea': 106}
_CORPUS = {'epic': EpicCorpus, 'egtea': EgteaCorpus}

parser = argparse.ArgumentParser(description=('Train language model from sequence of actions'))

# ------------------------------ Dataset -------------------------------
parser.add_argument('--train_pickle', type=Path)
parser.add_argument('--val_pickle', type=Path)
parser.add_argument('--verb_csv', type=Path, help='verb csv file if epic')
parser.add_argument('--noun_csv', type=Path, help='noun csv file if epic')
parser.add_argument('--action_csv', type=Path, help='action csv file if egtea')
parser.add_argument('--dataset', choices=['epic-55', 'epic-100', 'egtea'])
# ------------------------------ Model ---------------------------------
parser.add_argument('--num_gram', type=int, default=9)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--dim_feedforward', type=int, default=512)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=4)
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
parser.add_argument('--lr_steps', default=[25, 37], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=5, type=float,
                    metavar='W', help='gradient norm clipping')
# ------------------------------ Misc ------------------------------------
parser.add_argument('--output_dir', type=Path)
parser.add_argument('--disable_wandb_log', action='store_true')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=600, type=int,
                    metavar='N', help='print frequency (default: 10)')

args = parser.parse_args()

best_prec1 = 0
training_iterations = 0

if not args.output_dir.exists():
    args.output_dir.mkdir(parents=True)


def main():
    global args, best_prec1

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTCN_LM(_NUM_CLASSES[args.dataset], 
                    args.d_model, 
                    args.dim_feedforward, 
                    args.nhead,
                    args.num_layers, 
                    dropout=args.dropout)
    model = model.to(device)
    
    if not args.disable_wandb_log:
        wandb.init(project='MTCN', config=args)
        wandb.watch(model)

    if args.dataset.split('-')[0] == 'epic':
        csvfiles = [args.verb_csv, args.noun_csv]
    else:
        csvfiles = [args.action_csv]

    train_corpus = _CORPUS[args.dataset.split('-')[0]](args.train_pickle, csvfiles, _NUM_CLASSES[args.dataset], args.num_gram, train=True)
    val_corpus = _CORPUS[args.dataset.split('-')[0]](args.val_pickle, csvfiles, _NUM_CLASSES[args.dataset], args.num_gram, train=False)

    train_loader = torch.utils.data.DataLoader(
        train_corpus, 
        batch_size=args.batch_size, 
        shuffle=True, num_workers=args.workers, 
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_corpus, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=False)
    
    criterion = torch.nn.NLLLoss()

    # Optimizer and scheduler
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=args.lr, 
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=args.lr, 
                                     weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer, args.lr_steps, gamma=0.1)
    
    # Training loop
    for epoch in range(1, args.epochs):
        train(train_loader, model, criterion, epoch, optimizer, device)
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


def validate(val_loader, model, criterion, device, name=''):
    global training_iterations
    is_multitask = isinstance(model.num_class, list)
    ntokens = val_loader.dataset.num_class
    
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

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for batch, data in enumerate(val_loader):
            for key, item in data.items():
                data[key] = torch.transpose(item.to(device), 0, 1)
            
            if not is_multitask:
                output = model(data['input'])
                output = output.view(-1, ntokens)
                batch_size = output.size(0)
                output = F.log_softmax(output, dim=-1)

                loss = criterion(output, data['target'].reshape(-1))

                # Evaluate accuracies - Calculate accuracy only for masked positions
                output = output[data['input'].reshape(-1) == ntokens]
                target = data['target'][data['input'] == ntokens]
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
            else:
                output = model(data['verb_input'], data['noun_input'])
                output = output.view(-1, ntokens[0] + ntokens[1])
                batch_size = output.size(0)
                verb_output = F.log_softmax(output[..., :ntokens[0]], dim=-1)
                noun_output = F.log_softmax(output[..., ntokens[0]:], dim=-1)

                loss_verb = criterion(verb_output, data['verb_target'].reshape(-1))
                loss_noun = criterion(noun_output, data['noun_target'].reshape(-1))
                loss = 0.5 * (loss_verb + loss_noun)
                verb_losses.update(loss_verb.item(), batch_size)
                noun_losses.update(loss_noun.item(), batch_size)

                # Evaluate accuracies - Calculate accuracy only for masked positions
                verb_output = verb_output[data['verb_input'].reshape(-1) == ntokens[0]]
                noun_output = noun_output[data['noun_input'].reshape(-1) == ntokens[1]]
                verb_target = data['verb_target'][data['verb_input'] == ntokens[0]]
                noun_target = data['noun_target'][data['noun_input'] == ntokens[1]]
                
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

        # Logging    
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


def train(train_loader, model, criterion, epoch, optimizer, device):
    global training_iterations
    is_multitask = isinstance(model.num_class, list)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if is_multitask:
        verb_losses = AverageMeter()
        noun_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    ntokens = train_loader.dataset.num_class

    for i, data in enumerate(train_loader):
        for key, item in data.items():
            data[key] = torch.transpose(item.to(device), 0, 1)
            batch_size = data[key].size(0)

        ## Scheduled sampling - uncomment this if you want to use it
        data = scheduled_sampling(model, data, device, ntokens, p=0.2) if args.dataset == 'epic' else data

        if not is_multitask:
            output = model(data['input'])
            output = output.view(-1, ntokens)
            batch_size = output.size(0)

            output = F.log_softmax(output, dim=-1)

            loss= criterion(output, data['target'].reshape(-1))
        else:
            output = model(data['verb_input'], data['noun_input'])
            output = output.view(-1, ntokens[0] + ntokens[1])
            batch_size = output.size(0)

            verb_output = F.log_softmax(output[..., :ntokens[0]], dim=-1)
            noun_output = F.log_softmax(output[..., ntokens[0]:], dim=-1)

            loss_verb = criterion(verb_output, data['verb_target'].reshape(-1))
            loss_noun = criterion(noun_output, data['noun_target'].reshape(-1))
            loss = 0.5 * (loss_verb + loss_noun)
            verb_losses.update(loss_verb.item(), batch_size)
            noun_losses.update(loss_noun.item(), batch_size)
        losses.update(loss.item(), batch_size)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        training_iterations += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging    
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


def scheduled_sampling(model, data, device, ntokens, p=0.2):
    # This functino returns the scheduled sampling output with a certain probability p
    if random.uniform(0,1) < p:
        randomlist = torch.LongTensor([np.random.randint(0, args.num_gram - 1, size=2) for p in range(0, batch_size)]).to(device)
        temp_verbinput = torch.clone(data['verb_target'])
        temp_nouninput = torch.clone(data['noun_target'])
        batch_size = data['verb_target'].size(0)
        for ii in range(batch_size):
            temp_verbinput[randomlist[ii], ii] = ntokens[0]
            temp_nouninput[randomlist[ii], ii] = ntokens[1]
        
        with torch.no_grad():
            output_temp = model(temp_verbinput, temp_nouninput)
        
        verb_temp, noun_temp = [], []
        for ii in range(batch_size):
            verb_temp.append(torch.max(output_temp[randomlist[ii], ii, :ntokens[0]], dim=-1)[1])
            noun_temp.append(torch.max(output_temp[randomlist[ii], ii, ntokens[0]:], dim=-1)[1])

        for ii in range(batch_size):
            data['verb_input'][randomlist[ii], ii] = verb_temp[ii]
            data['noun_input'][randomlist[ii], ii] = noun_temp[ii]
    
    return data


if __name__ == '__main__':
    main()
