import torch
import math
import numpy as np
import shutil
from pathlib import Path


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).to(torch.float32).sum(0)
        res.append(float(correct_k.mul_(100.0 / batch_size)))
    return tuple(res)


def multitask_accuracy(outputs, labels, topk=(1,)):
    """
    Args:
        outputs: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        topk: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(topk))
    task_count = len(outputs)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    if torch.cuda.is_available():
        all_correct = all_correct.cuda(device=0)
    for output, label in zip(outputs, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    accuracies = []
    for k in topk:
        all_tasks_correct = torch.ge(all_correct[:k].float().sum(0), task_count)
        accuracy_at_k = float(all_tasks_correct.float().sum(0) * 100.0 / batch_size)
        accuracies.append(accuracy_at_k)
    return tuple(accuracies)


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pyth'):
    weights_dir = output_dir / Path('models')
    if not weights_dir.exists():
        weights_dir.mkdir(parents=True)
    torch.save(state, weights_dir / filename)
    if is_best:
        shutil.copyfile(weights_dir / filename,
                        weights_dir / 'model_best.pyth')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


##########################################################
##            BEAM SEARCH FUNCTIONS                     ##
##########################################################

def beam_search_decoder(predictions, top_k = 3):
    #start with an empty sequence with zero score
    output_sequences = [([], 0)]
    
    #looping through all the predictions
    for token_probs in predictions:
        new_sequences = []
        #append new tokens to old sequences and re-score
        for old_seq, old_score in output_sequences:
            for char_index in range(len(token_probs)):
                new_seq = old_seq + [char_index]
                #considering log-likelihood for scoring
                #new_score = old_score + math.log(token_probs[char_index])
                new_score = old_score + token_probs[char_index]
                new_sequences.append((new_seq, new_score))
                
        #sort all new sequences in the de-creasing order of their score
        output_sequences = sorted(new_sequences, key = lambda val: val[1], reverse = True)
        
        #select top-k based on score 
        # *Note- best sequence is with the highest score
        output_sequences = output_sequences[:top_k]
        
    return output_sequences


def get_topk(verb_sequence, noun_sequence, beam_size):
    # Conduct beam search on verb and noun individually - for Epic-kitchens
    verb_output = beam_search_decoder(verb_sequence, beam_size)
    noun_output = beam_search_decoder(noun_sequence, beam_size)

    return verb_output, noun_output


def get_topk_action(action_sequence, beam_size):
    # Conduct beam search on action - for EGTEA
    action_output = beam_search_decoder(action_sequence, beam_size)

    return action_output


def get_lmscore(verb_seq, noun_seq, model, num_gram, ntokens):
    # Calculate the LM score of the sequence for epic 
    verb_score, noun_score = 0, 0
    verb_input = verb_seq.repeat(1, num_gram)
    noun_input = noun_seq.repeat(1, num_gram)
    verb_input[range(num_gram), range(num_gram)] = ntokens[0]
    noun_input[range(num_gram), range(num_gram)] = ntokens[1]

    with torch.no_grad():
        output = model(verb_input, noun_input)
        verb_output = torch.nn.functional.log_softmax(output[..., :ntokens[0]], dim=-1)
        noun_output = torch.nn.functional.log_softmax(output[..., ntokens[0]:], dim=-1)
    verb_score = torch.sum(verb_output[range(num_gram), range(num_gram), verb_seq.reshape(-1)]).item()
    noun_score = torch.sum(noun_output[range(num_gram), range(num_gram), noun_seq.reshape(-1)]).item()

    return verb_score, noun_score


def get_lmscore_action(action_seq, model, num_gram, ntokens):
    # Calculate the LM score of the sequence for egtea
    action_score = 0
    action_input = action_seq.repeat(1, num_gram)
    action_input[range(num_gram), range(num_gram)] = ntokens
    
    with torch.no_grad():
        output = model(action_input, None)
        action_output = torch.nn.functional.log_softmax(output, dim=-1)
    action_score = torch.sum(action_output[range(num_gram), range(num_gram), action_seq.reshape(-1)]).item()

    return action_score
