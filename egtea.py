import torch
from torch.utils import data
import pandas as pd
import numpy as np
import h5py


class Egtea(data.Dataset):
    def __init__(self,
                 hdf5_path,
                 labels_pickle,
                 visual_feature_dim=2304,
                 audio_feature_dim=None,
                 window_len=5,
                 num_clips=10,
                 clips_mode='random',
                 labels_mode='center_action'):
        self.hdf5_dataset = None
        self.hdf5_path = hdf5_path
        self.df_labels = pd.read_pickle(labels_pickle)
        self.visual_feature_dim = visual_feature_dim
        self.window_len = window_len
        self.num_clips = num_clips
        assert clips_mode in ['all', 'random'], \
            "Labels mode not supported. Choose from ['all', 'random']"
        assert labels_mode in ['all', 'center_action'], \
            "Labels mode not supported. Choose from ['all', 'center_action']"
        self.clips_mode = clips_mode
        self.labels_mode = labels_mode

    def __getitem__(self, index):
        if self.hdf5_dataset is None:
            self.hdf5_dataset = h5py.File(self.hdf5_path, 'r')
        num_clips = self.num_clips if self.clips_mode == 'all' else 1
        data = torch.zeros((self.window_len * num_clips, self.visual_feature_dim))

        clip_name = self.df_labels.iloc[index]['clip_name']
        video_name = self.df_labels.iloc[index]['video_name']
        df_idx = self.df_labels.iloc[index].name
        df_sorted_video = self.df_labels[self.df_labels['video_name'] == video_name].sort_values('start_frame')
        idx = df_sorted_video.index.get_loc(df_idx)
        start = idx - self.window_len // 2
        end = idx + self.window_len // 2 + 1
        sequence_range = np.clip(np.arange(start, end), 0, df_sorted_video.shape[0] - 1)
        sequence_clip_names = df_sorted_video.iloc[sequence_range]['clip_name'].tolist()

        if self.clips_mode == 'random':
            for i in range(self.window_len):
                clip_idx = np.random.randint(self.num_clips)
                data[i] = torch.from_numpy(
                    self.hdf5_dataset['visual_features/' + str(sequence_clip_names[i])][clip_idx])
        else:
            for j in range(self.num_clips):
                for i in range(self.window_len):
                    data[i + j * self.window_len] = torch.from_numpy(
                        self.hdf5_dataset['visual_features/' + sequence_clip_names[i]][j])

        if self.labels_mode == "all":
            label = torch.from_numpy(df_sorted_video.iloc[sequence_range]['action_idx'].values)
            # Concatenate the labels of the center action in the end to be classified by the summary embedding
            label = torch.cat([label, label[self.window_len // 2].unsqueeze(0)])
        else:
            # Center action
            label = torch.tensor(df_sorted_video.iloc[idx]['action_idx'])

        return data, label, clip_name

    def __len__(self):
        return self.df_labels.shape[0]
