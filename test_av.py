import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix, accuracy_score

from epic_kitchens import EpicKitchens
from egtea import Egtea
from models_av import MTCN_AV
import pickle

_DATASETS = {'epic': EpicKitchens, 'egtea': Egtea}
_NUM_CLASSES = {'epic-55': [125, 352], 'epic-100': [97, 300], 'egtea': 106}


def eval_video(data, net, device):
    data = data.to(device)
    # For EGTEA, we feed each of 10 clips of each action in the sequence independently
    # to the audio-visual transformer and average their predictions, while for EPIC-KITCHENS
    # we feed all 10 clips for each action in the sequence simultaneously to the audio-visual transformer
    if args.dataset == 'egtea':
        data = data.view(10, -1, data.shape[2])
    if args.extract_attn_weights:
        rst, attn_weights = net(data, extract_attn_weights=args.extract_attn_weights)
    else:
        rst = net(data, extract_attn_weights=args.extract_attn_weights)
    if args.dataset == 'egtea':
        rst = torch.mean(rst, dim=0)

    if not isinstance(_NUM_CLASSES[args.dataset], list):
        if args.extract_attn_weights:
            return rst.cpu().numpy().squeeze(), attn_weights
        else:
            return rst.cpu().numpy().squeeze()
    else:
        if args.extract_attn_weights:
            return {'verb': rst[0].cpu().numpy().squeeze(),
                    'noun': rst[1].cpu().numpy().squeeze()},\
                    attn_weights
        else:
            return {'verb': rst[0].cpu().numpy().squeeze(),
                    'noun': rst[1].cpu().numpy().squeeze()}


def evaluate_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
    net = MTCN_AV(_NUM_CLASSES[args.dataset],
                  seq_len=args.seq_len,
                  num_clips=10 if 'epic' in args.dataset else 1,
                  visual_input_dim=args.visual_input_dim,
                  audio_input_dim=args.audio_input_dim if args.dataset.split('-')[0] == 'epic' else None,
                  d_model=args.d_model,
                  dim_feedforward=args.dim_feedforward,
                  nhead=args.nhead,
                  num_layers=args.num_layers,
                  dropout=args.dropout,
                  classification_mode='summary',
                  audio=not args.dataset == 'egtea')

    checkpoint = torch.load(args.checkpoint)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    net.load_state_dict(checkpoint['state_dict'])

    dataset = _DATASETS[args.dataset.split('-')[0]]
    test_loader = torch.utils.data.DataLoader(
        dataset(args.test_hdf5_path,
                args.test_pickle,
                visual_feature_dim=args.visual_input_dim,
                audio_feature_dim=args.audio_input_dim if args.dataset.split('-')[0] == 'epic' else None,
                window_len=args.seq_len,
                num_clips=10,
                clips_mode='all',),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    net = net.to(device)
    with torch.no_grad():
        net.eval()
        results = []
        if args.extract_attn_weights:
            attention_weights_dict = {}
        total_num = len(test_loader.dataset)

        proc_start_time = time.time()
        for i, (data, label, narration_id) in enumerate(test_loader):
            if args.extract_attn_weights:
                rst, attn_weights = eval_video(data, net, device)
            else:
                rst = eval_video(data, net, device)
            if not isinstance(_NUM_CLASSES[args.dataset], list):
                label_ = label.item()
            else:
                label_ = {k: v.item() for k, v in label.items()}
            results.append((rst, label_, narration_id))
            if args.extract_attn_weights:
                attention_weights_dict[narration_id[0]] = attn_weights
            cnt_time = time.time() - proc_start_time
            print('video {} done, total {}/{}, average {} sec/video'.format(
                i, i + 1, total_num, float(cnt_time) / (i + 1)))
        if args.extract_attn_weights:
            return results, attention_weights_dict
        else:
            return results


def print_accuracy(scores, labels):

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

    parser = argparse.ArgumentParser(description=('Test Audio-Visual Transformer on Sequence ' +
                                                  'of actions from untrimmed video'))
    parser.add_argument('--test_hdf5_path', type=Path)
    parser.add_argument('--test_pickle', type=Path)
    parser.add_argument('--dataset', choices=['epic-55', 'epic-100', 'egtea'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--visual_input_dim', type=int, default=2304)
    parser.add_argument('--audio_input_dim', type=int, default=2304)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--window_len', type=int, default=60)
    parser.add_argument('--extract_attn_weights', action='store_true')
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--split')
    parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    global args
    args = parser.parse_args()

    if args.extract_attn_weights:
        results, attention_weights_dict = evaluate_model()
    else:
        results = evaluate_model()
    if ('test' not in args.split and 'epic' in args.dataset) or 'epic' not in args.dataset:
        if isinstance(_NUM_CLASSES[args.dataset], list):
            keys = results[0][0].keys()
            for task in keys:
                print('Evaluation of {}'.format(task.upper()))
                print_accuracy([result[0][task] for result in results],
                               [result[1][task] for result in results])
        else:
            print_accuracy([result[0] for result in results],
                           [result[1] for result in results])

    output_dir = args.output_dir / Path('scores')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    save_scores(results, output_dir / Path(args.split+'.pkl'))

    if args.extract_attn_weights:
        attention_output_dir = args.output_dir / Path('attention')
        if not attention_output_dir.exists():
            attention_output_dir.mkdir(parents=True)
        attention_output_file = attention_output_dir / Path(args.split+'.pkl')
        with open(attention_output_file, 'wb') as f:
            pickle.dump(attention_weights_dict, f)


if __name__ == '__main__':
    main()
