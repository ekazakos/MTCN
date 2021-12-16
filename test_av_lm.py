import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.special import log_softmax
from collections import OrderedDict
import pandas as pd
import pickle

from models_lm import MTCN_LM
from utils import get_topk, get_topk_action, get_lmscore, get_lmscore_action

_NUM_CLASSES = {'epic-55': [125, 352], 'epic-100': [97, 300], 'egtea': 106}


def eval_epicvideos(av_results, df_labels, model, device, ntokens):
    # Beam search for verbs and nouns
    verb_scores = av_results['verb_output'].tolist()
    noun_scores = av_results['noun_output'].tolist()

    df_labels['verb_scores'] = verb_scores
    df_labels['noun_scores'] = noun_scores
    df_labels['verb_pred'] = ""
    df_labels['noun_pred'] = ""

    results = []

    video_ids = sorted(list(set(df_labels['video_id'])))

    for video_num, video_id in enumerate(video_ids):
        print("Processing video [{}/{}] ....".format(video_num + 1, len(video_ids)))
        df_video = df_labels[df_labels['video_id'] == video_id]
        df_video = df_video.sort_values(by='start_frame')
        
        for ii in range(len(df_video)):
            row = df_video.iloc[ii]
            verb_score, noun_score = torch.FloatTensor(row['verb_scores']).unsqueeze_(0), torch.FloatTensor(row['noun_scores']).unsqueeze_(0)
            
            narration_id = row.name

            verb_score = F.log_softmax(verb_score, dim=-1)
            noun_score = F.log_softmax(noun_score, dim=-1)

            if ii < args.num_gram // 2 or ii >= (len(df_video) - args.num_gram // 2):
                # Use the audio-visual output for corner actions
                verb_pred = verb_score.cpu().numpy().reshape(-1)
                noun_pred = noun_score.cpu().numpy().reshape(-1)
            else:
                verb_sequence = log_softmax(np.array(list(df_video['verb_scores'][ii - args.num_gram // 2: ii + args.num_gram // 2 + 1])), axis=-1)
                noun_sequence = log_softmax(np.array(list(df_video['noun_scores'][ii - args.num_gram // 2: ii + args.num_gram // 2 + 1])), axis=-1)
                
                verb_candidates, noun_candidates = get_topk(verb_sequence, noun_sequence, args.beam_size)
         
                verb_pred = verb_score.cpu().numpy().reshape(-1)
                noun_pred = noun_score.cpu().numpy().reshape(-1)
                verb_dict = {}
                noun_dict = {}

                # Beam search
                for jj in range(args.beam_size):
                    for kk in range(args.beam_size):
                        verb_input, verb_avscore = verb_candidates[jj]
                        noun_input, noun_avscore = noun_candidates[kk]
                        verb_input = torch.LongTensor(verb_input).unsqueeze_(1).to(device)
                        noun_input = torch.LongTensor(noun_input).unsqueeze_(1).to(device)
                        verb_lmscore, noun_lmscore = get_lmscore(verb_input, noun_input, model, args.num_gram, ntokens)

                        # LM fusion with hyperparameter alpha
                        verb_score = (1 - args.alpha) * verb_avscore + args.alpha * verb_lmscore
                        noun_score = (1 - args.alpha) * noun_avscore + args.alpha * noun_lmscore

                        verb_center = verb_candidates[jj][0][args.num_gram // 2]
                        noun_center = noun_candidates[kk][0][args.num_gram // 2]

                        if verb_center not in verb_dict:
                            verb_dict[verb_center] = verb_score
                        if noun_center not in noun_dict:
                            noun_dict[noun_center] = noun_score
                        if verb_dict[verb_center] < verb_score:
                            verb_dict[verb_center] = verb_score
                        if noun_dict[noun_center] < noun_score:
                            noun_dict[noun_center] = noun_score
       
                verb_dict = OrderedDict([(k,v) for k, v in sorted(verb_dict.items(), key=lambda item: item[1], reverse=False)])
                noun_dict = OrderedDict([(k,v) for k, v in sorted(noun_dict.items(), key=lambda item: item[1], reverse=False)])
                verb_max = np.max(verb_pred)
                noun_max = np.max(noun_pred)

                c = 0.1
                for jj, (key, item) in enumerate(verb_dict.items()):
                    verb_pred[key] = verb_max + c * (jj + 1)
                for jj, (key, item) in enumerate(noun_dict.items()):
                    noun_pred[key] = noun_max + c * (jj + 1)

            df_labels.at[narration_id, 'verb_pred'] = verb_pred
            df_labels.at[narration_id, 'noun_pred'] = noun_pred

    for ii in range(len(df_labels)):
        row = df_labels.iloc[ii]
        rst_ = {'verb': row['verb_pred'], 'noun' : row['noun_pred']}
        labels_ = {'verb' : row['verb_class'], 'noun' : row['noun_class']} if args.split != 'test' else {}
        narration_id = row.name
        results.append((rst_, labels_, narration_id))

    return results


def eval_egteavideos(av_results, df_labels, model, device, ntokens):
    # Beam search for actions 
    action_scores = av_results['scores'].tolist()
    action_classes = av_results['labels'].tolist()

    df_labels['action_scores'] = action_scores
    df_labels['action_class'] = action_classes
    df_labels['action_pred'] = ""

    results = []

    video_ids = sorted(list(set(df_labels['video_name'])))

    for video_num, video_id in enumerate(video_ids):
        print("Processing video [{}/{}] ....".format(video_num + 1, len(video_ids)))
        df_video = df_labels[df_labels['video_name'] == video_id]
        df_video = df_video.sort_values(by='start_frame')
        
        for ii in range(len(df_video)):
            row = df_video.iloc[ii]
            action_score = torch.FloatTensor(row['action_scores']).unsqueeze_(0)
            action_score = F.log_softmax(action_score, dim=-1)
            narration_id = row.name

            if ii < args.num_gram // 2 or ii >= (len(df_video) - args.num_gram // 2):
                # Use the audio-visual output for corner actions
                action_pred = action_score.cpu().numpy().reshape(-1)
            else:
                action_sequence = log_softmax(np.array(list(df_video['action_scores'][ii - args.num_gram // 2: ii + args.num_gram // 2 + 1])), axis=-1)
                action_candidates = get_topk_action(action_sequence, args.beam_size)
         
                action_pred = action_score.cpu().numpy().reshape(-1)
                action_dict = {}

                # Beam search
                for jj in range(args.beam_size):
                    action_input, action_avscore = action_candidates[jj]
                    action_input = torch.LongTensor(action_input).unsqueeze_(1).to(device)
                    action_lmscore = get_lmscore_action(action_input, model, args.num_gram, ntokens)

                    # LM fusion with hyperparameter alpha
                    action_score = (1 - args.alpha) * action_avscore + args.alpha * action_lmscore
                    action_center = action_candidates[jj][0][args.num_gram // 2]

                    if action_center not in action_dict:
                        action_dict[action_center] = action_score
                    if action_dict[action_center] < action_score:
                        action_dict[action_center] = action_score
       
                action_dict = OrderedDict([(k,v) for k, v in sorted(action_dict.items(), key=lambda item: item[1], reverse=False)])
                action_max = np.max(action_pred)

                c = 0.1
                for jj, (key, item) in enumerate(action_dict.items()):
                    action_pred[key] = action_max + c * (jj + 1)

            df_labels.at[narration_id, 'action_pred'] = action_pred

    for ii in range(len(df_labels)):
        row = df_labels.iloc[ii]
        rst_ = row['action_pred']
        labels_ = row['action_class']
        narration_id = row.name
        results.append((rst_, labels_, narration_id))

    return results


def print_accuracy(scores, labels):
    # Printing accuracy and average per-class accuracy
    video_pred = [np.argmax(score) for score in scores]
    cf = confusion_matrix(labels, video_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_cnt[cls_hit == 0] = 1  # to avoid divisions by zero
    cls_acc = cls_hit / cls_cnt

    acc = accuracy_score(labels, video_pred)

    print('Accuracy {:.02f}%'.format(acc * 100))
    print('Average Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))


def save_scores(results, output):
    # Save the scores as a pickle format
    save_dict = {}
    if not isinstance(_NUM_CLASSES[args.dataset], list):
        scores = np.array([result[0] for result in results])
        labels = np.array([result[1] for result in results])
        save_dict['scores'] = scores
        save_dict['labels'] = labels
    else:
        keys = results[0][0].keys()
        save_dict = {k + '_output': np.array([result[0][k] for result in results]) for k in keys}
        save_dict['narration_id'] = np.array([result[2] for result in results])

    with open(output, 'wb') as f:
        pickle.dump(save_dict, f)


def main():
    parser = argparse.ArgumentParser(description=('Fuse the MTCN output scores and LM scores'))
    parser.add_argument('--test_pickle', type=Path)
    parser.add_argument('--test_scores', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--dataset', choices=['epic-55', 'epic-100', 'egtea'])
    parser.add_argument('--num_gram', default=9, type=int)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    # ------------------------------ BEAM SEARCH ----------------------------------
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--beam_size', type=int, default=10)
    # ------------------------------ OUTPUT ----------------------------------
    parser.add_argument('--output_dir', type=Path, default='scores')
    parser.add_argument('--split', type=str, default='result')

    global args
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = _NUM_CLASSES[args.dataset]

    # Load model
    model = MTCN_LM(ntokens, 
                args.d_model, 
                args.dim_feedforward, 
                args.nhead,
                args.num_layers)
                
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    model = model.to(device)
    model.eval()
    
    # For beam search
    if args.alpha == 0:
        # You don't need a beam search for this
        args.beam_size = 1
    
    # Load the audio-visual output
    with open(args.test_scores, 'rb') as f:
        av_results = pickle.load(f)
    
    df_labels = pd.read_pickle(args.test_pickle)

    if args.dataset.split('-')[0] == 'epic':
        results = eval_epicvideos(av_results, df_labels, model, device, ntokens)
    else:
        results = eval_egteavideos(av_results, df_labels, model, device, ntokens)

    print("ALPHA : {}, BEAM_SIZE : {}".format(args.alpha, args.beam_size))

    # Print accuracy
    if ('test' not in args.split and 'epic' in args.dataset) or 'epic' not in args.dataset:
        if isinstance(ntokens, list):
            keys = results[0][0].keys()
            for task in keys:
                print('Evaluation of {}'.format(task.upper()))
                print_accuracy([result[0][task] for result in results],
                            [result[1][task] for result in results])
        else:
            print_accuracy([result[0] for result in results],
                        [result[1] for result in results])

    # Save the scores file
    output_dir = args.output_dir / Path('scores')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    save_scores(results, output_dir / Path(args.split + '.pkl'))


if __name__ == '__main__':
    main()