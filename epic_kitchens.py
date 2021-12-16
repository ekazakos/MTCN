import torch
from torch.utils import data
import pandas as pd
import numpy as np
import h5py


class EpicKitchens(data.Dataset):
    def __init__(self,
                 hdf5_path,
                 labels_pickle,
                 visual_feature_dim=2304,
                 audio_feature_dim=2304,
                 window_len=5,
                 num_clips=10,
                 clips_mode='random',
                 labels_mode='center_action'):
        self.hdf5_dataset = None
        self.hdf5_path = hdf5_path
        self.df_labels = pd.read_pickle(labels_pickle)
        self.visual_feature_dim = visual_feature_dim
        self.audio_feature_dim = audio_feature_dim
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
        data = torch.zeros((2 * self.window_len * num_clips, max(self.visual_feature_dim, self.audio_feature_dim)))

        narration_id = self.df_labels.iloc[index].name
        video_id = self.df_labels.iloc[index]['video_id']
        df_sorted_video = self.df_labels[self.df_labels['video_id'] == video_id].sort_values('start_timestamp')
        idx = df_sorted_video.index.get_loc(narration_id)
        start = idx - self.window_len // 2
        end = idx + self.window_len // 2 + 1
        sequence_range = np.clip(np.arange(start, end), 0, df_sorted_video.shape[0] - 1)
        sequence_narration_ids = df_sorted_video.iloc[sequence_range].index.tolist()

        if self.clips_mode == 'random':
            for i in range(self.window_len):
                clip_idx = np.random.randint(self.num_clips)
                data[i][:self.visual_feature_dim] = torch.from_numpy(
                    self.hdf5_dataset['visual_features/' + str(sequence_narration_ids[i])][clip_idx])
                data[self.window_len + i][:self.audio_feature_dim] = torch.from_numpy(
                    self.hdf5_dataset['audio_features/' + str(sequence_narration_ids[i])][clip_idx])
        else:
            for i in range(self.window_len):
                for j in range(self.num_clips):
                    data[i * self.num_clips + j][:self.visual_feature_dim] = torch.from_numpy(
                        self.hdf5_dataset['visual_features/' + str(sequence_narration_ids[i])][j])
                    data[self.window_len * self.num_clips + i * self.num_clips + j][:self.audio_feature_dim] = torch.from_numpy(
                        self.hdf5_dataset['audio_features/' + str(sequence_narration_ids[i])][j])

        if self.labels_mode == "all":
            verbs = torch.from_numpy(df_sorted_video.iloc[sequence_range]['verb_class'].values) \
                if 'verb_class' in df_sorted_video.columns else torch.full((self.window_len,), -1)
            nouns = torch.from_numpy(df_sorted_video.iloc[sequence_range]['noun_class'].values) \
                if 'noun_class' in df_sorted_video.columns else torch.full((self.window_len,), -1)
            # Replicate sequence of labels x2, 1 for video sequence and 1 audio sequence
            verbs = verbs.repeat(2)
            nouns = nouns.repeat(2)
            # Concatenate the labels of the center action in the end to be classified by the summary embedding
            verbs = torch.cat([verbs, verbs[self.window_len // 2].unsqueeze(0)])
            nouns = torch.cat([nouns, nouns[self.window_len // 2].unsqueeze(0)])
            label = {'verb': verbs, 'noun': nouns}
        else:
            # Center action
            verb = torch.tensor(df_sorted_video.iloc[idx]['verb_class']) \
                if 'verb_class' in df_sorted_video.columns else torch.full((1,), -1)
            noun = torch.tensor(df_sorted_video.iloc[idx]['noun_class']) \
                if 'noun_class' in df_sorted_video.columns else torch.full((1,), -1)
            label = {'verb': verb, 'noun': noun}

        return data, label, narration_id

    def __len__(self):
        return self.df_labels.shape[0]
