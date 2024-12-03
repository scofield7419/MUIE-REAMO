
from dataclasses import dataclass, field
from torch.utils.data import Dataset, IterableDataset
from training_utils import DataArguments
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import torch 
import torch.nn as nn
import os
from PIL import Image
import copy
from .dataset_utils import *
import numpy as np
import random
import cv2
from diffusers.image_processor import VaeImageProcessor
from diffusers.video_processor import VideoProcessor
from .audio_processor import VaeAudioProcessor


def order_pick_k(lst, k):
    if len(lst) <= k:
        return lst
    rng = np.random.random(len(lst))
    index = np.argsort(rng)[:k]
    index_sort = sorted(index)
    new_lst = [lst[i] for i in index_sort]
    print(
        f"WARNING: total file: {len(lst)}, random pick: {k}."
        f" (ignored)"
    )
    return new_lst


def read_video(video_path, sample_fps=1, max_frames=8, height=320, width=576, get_first_frame=False):
    """
    Read video frames from video_path.
    Args:
        video_path: str, path to the video file.
        sample_fps: int, sample frames per second.
        max_frames: int, maximum number of frames to sample.
    Returns:
        torch.Tensor, (num_frames, channel, height, width).
    """
    height = 0
    width = 0
    for _ in range(5):
        try:
            capture = cv2.VideoCapture(video_path)
            _fps = capture.get(cv2.CAP_PROP_FPS)
            # print(_fps)
            _total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            # print(_total_frame_num)
            stride = round(_fps / sample_fps)
            cover_frame_num = (stride * max_frames)
            # print(cover_frame_num)
            if _total_frame_num < cover_frame_num + 5:
                start_frame = 0
                end_frame = _total_frame_num
            else:
                start_frame = random.randint(0, _total_frame_num-cover_frame_num-5)
                end_frame = start_frame + cover_frame_num
            
            pointer, frame_list = 0, []
            while(True):
                ret, frame = capture.read()
                pointer +=1 
                if (not ret) or (frame is None): break
                if pointer < start_frame: continue
                if pointer >= end_frame - 1: break
                if (pointer - start_frame) % stride == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    height, width = frame.size
                    frame_list.append(frame)
            break
        except Exception as e:
            print('{} read video frame failed with error: {}'.format(video_path, e))
            continue
    
    assert height > 0 and width > 0, "Video height and width should be greater than 0."
    # video_data = 

    dummy_frame = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    try:
        if len(frame_list)> max_frames:
            # mid_frame = copy(frame_list[ref_idx])
            # vit_frame = self.vit_transforms(mid_frame)
            # frames = self.transforms(frame_list)
            frame_list = frame_list[:max_frames]
        elif 0< len(frame_list) < max_frames:
            frame_list.extend([dummy_frame] * (max_frames - len(frame_list)))
        else:
            pass
    except Exception as e:
        print('{} read video frame failed with error: {}'.format(video_path, e))
    
    return frame_list


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample or 'video' in sample or 'audio' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # print("Loading data from ", i)
        # print("Sources: ", sources)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'input_image' in sources[0]:
            image_file = self.list_data_dict[i]['input_image']
            image_folder = self.data_args.image_folder
            # print("Loading image from ", image_folder)
            processor = self.data_args.image_processor
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image_file = image_file if isinstance(image_file, list) else [image_file]
            image_file = order_pick_k(image_file, 8)
            if image_folder is None:
                try:
                    image = [Image.open(file).convert('RGB') for file in image_file]
                except:
                    print("Image file: ", image_file)
                    print(self.list_data_dict[i])
                    print("Image folder: ", image_folder)
                    exit()
            else:
                image = [Image.open(os.path.join(image_folder, file)).convert('RGB') for file in image_file]
                
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = [expand2square(i, tuple(int(x*255) for x in processor.image_mean)) for i in image]
                
                image = [processor(images=i, return_tensors="pt")['pixel_values'] for i in image]
            else:
                image = [processor(images=i, return_tensors='pt')['pixel_values'] for i in image]
    
        
        if 'input_video' in sources[0]:
            video_file = self.list_data_dict[i]['input_video']
            video_folder = self.data_args.video_folder
            # print("Loading video from ", video_folder)
            processor = self.data_args.video_processor
            # video = os.path.join(video_file, video_folder)
            video_file = video_file if isinstance(video_file, list) else [video_file]
            video_file = order_pick_k(video_file, 8)
            if video_folder is None:
                video = video_file
            else:
                video = [os.path.join(video_folder, file) for file in video_file]
            video = processor(videos=video, return_tensors='pt')['pixel_values']

        if 'input_audio' in sources[0]:
            audio_file = self.list_data_dict[i]['input_audio']
            audio_folder = self.data_args.audio_folder
            # print("Loading audio from ", audio_folder)
            processor = self.data_args.audio_processor
            # audio = os.path.join(audio_folder, audio_file)
            audio_file = audio_file if isinstance(audio_file, list) else [audio_file]
            audio_file = order_pick_k(audio_file, 8)
            if audio_folder is None:
                audio = audio_file
            else:
                audio = [os.path.join(audio_folder, file) for file in audio_file]
            audio = processor(audios=audio, return_tensors='pt')['pixel_values']

        # if 'image' in sources[0] or 'video' in sources[0] or 'audio' in sources[0]:
        sources = preprocess_multimodal(
            copy.deepcopy([e["conversations"] for e in sources]),
            self.data_args)
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_other_modality=True)  # ('image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i] or 'audio' in self.list_data_dict[i])
        # print("Data dict: ", data_dict)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'input_image' in self.list_data_dict[i]:
            data_dict['image'] = image
        
        if 'input_video' in self.list_data_dict[i]:
            data_dict['video'] = video

        if 'input_audio' in self.list_data_dict[i]:
            data_dict['audio'] = audio

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
                                                    input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :1024]
        labels = labels[:, :1024]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        new_images = []
        new_audios = []
        new_videos = []

        for instance in instances:
            if 'image' in instance:
                if type(instance['image']) is list:
                    for i in instance['image']:
                        new_images.append(i)
                else:
                    new_images.append(instance['image'])

            if 'video' in instance:
                if type(instance['video']) is list:
                    for i in instance['video']:
                        new_videos.append(i)
                else:
                    new_videos.append(instance['video'])
            
            if 'audio' in instance:
                if type(instance['audio']) is list:
                    for i in instance['audio']:
                        new_audios.append(i)
                else:
                    new_audios.append(instance['audio'])
        
        if len(new_images) > 0:
            batch['images'] = new_images
        
        if len(new_videos) > 0:
            batch['videos'] = new_videos
        
        if len(new_audios) > 0:
            batch['audios'] = new_audios
    
        return batch
    



