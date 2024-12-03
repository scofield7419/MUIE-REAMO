import os


class DatasetCatalog:
    def __init__(self):
        # the following dataset utilized for alignment learning
        self.audiocap_enc = {
            "target": "reamo.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/T_X_pair_data/audiocap/audiocap_comprehension.json",
                audio_folder="./data/T_X_pair_data/audiocap/audios",
                image_folder="./data/T_X_pair_data/cc3m/images",
                video_folder="./data/T_X_pair_data/webvid/videos",
            ),
        }

        self.webvid_enc = {
            "target": "reamo.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/T_X_pair_data/webvid/webvid_comprehension.json",
                video_folder="./data/T_X_pair_data/webvid/videos",
                audio_folder="./data/T_X_pair_data/audiocap/audios",
                image_folder="./data/T_X_pair_data/cc3m/images",
            ),
        }

        self.cc3m_enc = {
            "target": "reamo.dataset.base_dataset.LazySupervisedDataset",
            "params": dict(
                data_path="./data/T_X_pair_data/cc3m/cc3m_comprehension.json",
                video_folder="./data/T_X_pair_data/webvid/videos",
                audio_folder="./data/T_X_pair_data/audiocap/audios",
                image_folder="./data/T_X_pair_data/cc3m/images",
            ),
        }

        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
