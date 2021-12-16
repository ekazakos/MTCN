import pandas as pd
import numpy as np
import torch
import random

from torch.utils.data import Dataset

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.idx2count = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.idx2count[len(self.idx2word) - 1] = 0
        return self.word2idx[word]

    def add_count(self, idx):
        self.idx2count[idx] += 1

    def __len__(self):
        return len(self.idx2word)


class EpicCorpus(Dataset):
    def __init__(self, pickle_file, csvfiles, num_class, num_gram, train=True):
        self.verb_dict, self.noun_dict = Dictionary(), Dictionary()
        verb_csv, noun_csv = csvfiles[0], csvfiles[1]
        self.num_class = num_class
        self.num_gram = num_gram
        self.train = train
        
        assert num_gram >= 2

        # Update verb & noun dictionary, note that last token is '<mask>' token
        with open(verb_csv, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                idx, word = int(line.split(',')[0]), line.split(',')[1]
                self.verb_dict.add_word(word)
        self.verb_dict.add_word('<mask>')
        
        with open(noun_csv, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                idx, word = int(line.split(',')[0]), line.split(',')[1]
                self.noun_dict.add_word(word)
        self.noun_dict.add_word('<mask>')
        self.verbs, self.nouns = self.tokenize(pd.read_pickle(pickle_file))

    def tokenize(self, df_labels):
        """Tokenizes a epic-kitchens file."""
        # Parse the pandas file
        video_ids = sorted(list(set(df_labels['video_id'])))
        verb_idss, noun_idss = [], []

        for video_id in video_ids:
            df_video = df_labels[df_labels['video_id'] == video_id]
            df_video = df_video.sort_values(by='start_frame')
            verb_class = list(df_video['verb_class'])
            noun_class = list(df_video['noun_class'])

            for verbidx in verb_class:
                self.verb_dict.add_count(verbidx)
            for nounidx in noun_class:
                self.noun_dict.add_count(nounidx)

            assert len(verb_class) == len(noun_class)
            for ii in range(len(verb_class) - self.num_gram + 1):
                verb_temp = []
                noun_temp = []
                for j in range(self.num_gram):
                    verb_temp.append(verb_class[ii + j])
                    noun_temp.append(noun_class[ii + j])
                verb_idss.append(torch.tensor(verb_temp).type(torch.int64))
                noun_idss.append(torch.tensor(noun_temp).type(torch.int64))

        verb_ids = torch.stack(verb_idss, dim=0)
        noun_ids = torch.stack(noun_idss, dim=0)
        
        assert verb_ids.shape[0] == noun_ids.shape[0]
        return verb_ids, noun_ids

    def __len__(self):
        return len(self.verbs)
    
    def __getitem__(self, index):
        verb, noun = self.verbs[index], self.nouns[index]
        verb_input, noun_input = verb.clone().detach(), noun.clone().detach()

        if self.train:
            verb_mask_pos = np.random.choice(list(range(self.num_gram)))
            noun_mask_pos = verb_mask_pos

            verb_input[verb_mask_pos] = self.verb_dict.word2idx['<mask>']
            noun_input[noun_mask_pos] = self.noun_dict.word2idx['<mask>']
            
        else:
            # For evaluating, test only the centre action
            mask_pos = self.num_gram // 2
            verb_input[mask_pos] = self.verb_dict.word2idx['<mask>']
            noun_input[mask_pos] = self.noun_dict.word2idx['<mask>']

        data = {'verb_input': verb_input, 'verb_target': verb, 'noun_input': noun_input, 'noun_target' : noun}
        return data
        

class EgteaCorpus(Dataset):
    def __init__(self, pickle_file, csvfiles, num_class, num_gram, train=True):
        self.action_dict = Dictionary()
        self.num_class = int(num_class)
        self.num_gram = num_gram
        self.train = train
        action_csv = csvfiles[0]

        assert num_gram >= 2
        
        # Update action dictionary, note that last token is '<mask>' token
        with open(action_csv, 'r') as f:
            lines = f.readlines()
            for line in lines:
                idx, word = int(line.split(',')[0]), line.split(',')[1]
                self.action_dict.add_word(word)
        self.action_dict.add_word('<mask>')
        self.actions = self.tokenize(pd.read_pickle(pickle_file))

    def tokenize(self, df_labels):
        """Tokenizes a epic-kitchens file."""
        # Parse the pandas file
        video_ids = sorted(list(set(df_labels['video_name'])))
        action_idss = []

        for video_id in video_ids:
            df_video = df_labels[df_labels['video_name'] == video_id]
            df_video = df_video.sort_values(by='start_frame')
            action_class = list(df_video['action_idx'])

            for actionidx in action_class:
                self.action_dict.add_count(actionidx)

            for ii in range(len(action_class) - self.num_gram + 1):
                action_temp = []
                for j in range(self.num_gram):
                    action_temp.append(action_class[ii + j])
                action_idss.append(torch.tensor(action_temp).type(torch.int64))
            
        action_ids = torch.stack(action_idss, dim=0)
        
        return action_ids

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        action = self.actions[index]
        action_input = action.clone().detach()

        if self.train:
            mask_pos = np.random.choice(list(range(self.num_gram)))
            action_input[mask_pos] = self.action_dict.word2idx['<mask>']

        else:
            # For evaluating, test only the centre action
            mask_pos = self.num_gram // 2
            action_input[mask_pos] = self.action_dict.word2idx['<mask>']

        data = {'input': action_input, 'target': action}
        return data
        
        