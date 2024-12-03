import torch

from reamo.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN, DEFAULT_VIDEO_TOKEN
from reamo.conversation import conv_templates, SeparatorStyle
from reamo.model.builder import load_pretrained_model
from reamo.utils import disable_torch_init
from reamo.mm_utils import tokenizer_image_token, tokenizer_multiple_token
from transformers.generation.streamers import TextIteratorStreamer
import transformers
from dataclasses import dataclass, field
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import List
from PIL import Image
import requests
from io import BytesIO
import scipy
from cog import BasePredictor, Input, Path, ConcatenateIterator
import time
import subprocess
from threading import Thread
from diffusers.utils import export_to_video
import os
import re
import sys
import cv2
import tempfile
import numpy as np


os.environ["BASE_HOME"] = os.path.dirname(__file__)
sys.path.append(os.path.dirname(__file__))

sys.path.append(os.path.join(os.environ['BASE_HOME'], 'modules/SEEM/demo_code'))
sys.path.append(os.path.join(os.environ['BASE_HOME'], 'modules/SEEM/demo_code/tasks'))
import modules.SEEM.demo_code.app as SEEM
import modules.SEEM.demo_code.utils.visualizer as visual

sys.path.append(os.path.join(os.environ['BASE_HOME'], 'modules/SHAS'))
import modules.SHAS.segment as audio_segment


def open_image(image):
    if type(image) == np.ndarray:
        image = Image.fromarray(image)
    elif type(image) == str:
        image = Image.open(image).convert("RGB")
    return image


def save_image_to_local(image: Image.Image):
    # TODO: Update so the url path is used, to prevent repeat saving.
    if not os.path.exists('temp'):
        os.mkdir('temp')
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image.save(filename)
    return filename

@dataclass
class GenerateArguments:
    # Basic generation arguments
    top_k: int = field(default=1, metadata={"help": "The number of highest probability tokens to keep for top-k-filtering in the sampling strategy"})
    top_p: float = field(default=1.0, metadata={"help": "The cumulative probability for top-p-filtering in the sampling strategy."})
    temperature: float = field(default=1.0, metadata={"help": "The value used to module the next token probabilities. Must be strictly positive."},)
    max_new_tokens: int = field(default=100, metadata={"help": "The maximum number of new tokens to generate. The generation process will stop when reaching this threshold."})
    do_sample: bool = field(default=True, metadata={"help": "Whether to sample from the output distribution to generate new tokens. If False, use argmax."})
    use_cache: bool = field(default=False, metadata={"help": "Whether to cache the hidden states of the model to speed up generation."})
    output_hidden_states: bool = field(default=True,metadata={"help": "Whether to return the hidden states of all intermediate layers."})
    

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            _stop = torch.tensor(stop).to(input_ids[0].device)
            indices = torch.where(_stop[0] == input_ids)
            for i in indices:
                if len(i) > 0:
                    if torch.all(input_ids[0][i:i + len(_stop)] == _stop):
                        stop_count += 1
        if stop_count >= self.ENCOUNTERS:
            return True
        return False


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # for weight in weights:
        #     download_weights(weight["src"], weight["dest"], weight["files"])
        disable_torch_init()

        # /home/haofei/pretrain_ckpt/vicuna-7b-v1.5
        self.tokenizer, self.model, self.image_processor, self.video_processor, self.audio_processor, self.context_len, self.model_config = load_pretrained_model(model_base="./checkpoints/pretrain_enc_2/model", 
                                                                                                   model_name="reamo-v1.5-7b", 
                                                                                                   model_path="./checkpoints/finetune_3", 
                                                                                                   load_8bit=False, load_4bit=False)

    def find_module_content(self, data):
        pattern = r'<Module>(.*?)</Module>'
        match = re.search(pattern, data)
        if match:
            return match.group(1)
        else:
            return None


    def find_instruction_content(self, data):
        pattern = r'<Instruction>(.*?)</Instruction>'
        match = re.findall(pattern, data)
        
        if match:
            res = []
            for _res in match:
                res.append(_res.split(':')[-1].strip())
            return res
        else:
            return None

    def remove_special_tags(self, text):
        """
        remove the content between the tags and also the tags: <module></module> <instruction></instruction> <region></region> <SP></SP>
        """
        pattern = r'<[^>]+>(.*?)<[^>]+>'  # match all the tags
        return re.sub(pattern, '', text)


    def parse_model_output(self, model_output):
        """
        Based on the model output, we parse the model output and return the parsed instructions.
        Args:
            model_output (str): The model output.
        """
        # Parse the model output
        module = self.find_module_content(model_output)
        instruction = self.find_instruction_content(model_output)
        output = self.remove_special_tags(model_output)
        return output, module, instruction

    

    def audio_segmentation(self, audio_path, path_to_checkpoint, path_to_segmentation_yaml):
        """
        Based on the input audio, we segment the audio and return the segmented audio.
        Args:
            audio_path (str): The input audio path.
            track_text (str): The reference text.
        Returns:
            str: The segmented audio path.
        """
        if audio_path is None:
            return None
        audio_segment.segment(path_to_wavs=audio_path, path_to_checkpoint=path_to_checkpoint, path_to_segmentation_yaml=path_to_segmentation_yaml)
        return path_to_segmentation_yaml

    def video_tracking(Self, video_path=None, sketch_pad=None, track_prompt="", text_prompt=""):
        """
        Based on the input video, we track the video and return the tracked video.
        Args:
            video_path (str): The input video path.
            track_text (str): The track text.
            sketch_pad (dict): The sketchpad input with format {'image': Image, 'mask': Image}.
            track_prompt (str): The track prompt.
            text_prompt (str): The text prompt.  
                if no sketchpad, the text prompt is used to segment the image, obtaining the foreground images.
        Returns:
            str: The tracked video path.
        """
        if video_path is None:
            return None
        if sketch_pad is None:
            i_video_path = video_path.split('/')[-2]
            img, o = video_editing(video_path=i_video_path, fore_prompt=text_prompt, back_prompt="")
            image_path = save_image_to_local(img)
            img = Image.open(image_path)
            compose_img = {'image': img, 'mask': img}
        else:
            # compose_img = sketch_pad
            compose_img = {'image': open_image(sketch_pad['image']), 'mask': sketch_pad['mask']}
            # compose_img = {'image': sketch_pad['ibs'].image, 'mask':  sketch_pad['ibs'].masks[-1]}
            # print(save_image_to_local(open_image(sketch_pad['image'])))  # an image with bbox
            # print(save_image_to_local(open_image(sketch_pad['mask'])))  # a binary mask with strech

        _, output_video_name = SEEM.inference("examples/placeholder.png", task=['Video'],
                                            video_pth=video_path, refimg=compose_img, reftxt=track_prompt)
        return output_video_name

    def image_segmentation(self, image_path, track_text, sketch_pad=None):
        """
        Based on the input image, we segment the image and return the segmented image.
        Args:
            image (Image): The input image.
            track_text (str): The reference text.
            sketch_pad (Dict):
                ['image']: array
                ['mask']: array
        Returns:
            Image: The segmented image.
        """
        print('Calling SEEM_app.inference')
        if image_path is None:
            return None, None
        img = open_image(image_path)
        width, height = img.size
        if len(track_text) == 0 and sketch_pad is None:
            # segment all
            compose_img = {'image': img, 'mask': img}
            task = []
            image, _, labels = SEEM.inference(image=compose_img, task=task, reftxt=track_text)
            return image[0], _, labels
        if sketch_pad is not None:
            compose_img = {'image': open_image(sketch_pad['image']), 'mask': sketch_pad['image']}
            # print('mask path: ', save_image_to_local(open_image(sketch_pad['image'])))
            # print('image segmentation / sketch_pad', sketch_pad)  # sketch_pad['image']: array,  sketch_pad['mask']: array
            width, height = compose_img['image'].width, compose_img['image'].height
            task = ['Stroke']
        else:
            compose_img = {'image': img, 'mask': img}
            task = ['Text']
        
        image, masks, labels = SEEM.inference(image=compose_img, task=task, reftxt=track_text)
        mask_pred = masks[0].astype("uint8")
        mask_pred = cv2.resize(mask_pred, (width, height), interpolation=cv2.INTER_LANCZOS4)
        mask_pred = mask_pred.astype("uint8")
        print('mask_pred: ', mask_pred)
        mask_demo = visual.GenericMask(mask_pred, height, width)
        bbox = mask_demo.bbox()
        mask = {'mask': mask_pred, 'boxes': bbox}
        return image[0], mask, labels

    def predict(
        self,
        image: str = None,
        video: str = None,
        audio: str = None,
        prompt: str = None,
        top_p: float = 1.0,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
    ):
        """Run a single prediction on the model"""

        # prepare generation arguments
        parser = transformers.HfArgumentParser(GenerateArguments)
        generation_args = parser.parse_args_into_dataclasses()[0]

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[[835]], encounters=1)])

        generation_args.top_p = top_p if top_p is not None else generation_args.top_p
        generation_args.temperature = temperature if temperature is not None else generation_args.temperature
        generation_args.max_new_tokens = max_new_tokens if max_new_tokens is not None else generation_args.max_new_tokens
        generation_args.stopping_criteria = stopping_criteria

        conv_mode = "vicuna_v1"  # conv_llava_plain  conv_vicuna_v1
        conv = conv_templates[conv_mode].copy()

        image_tensor = None
        video_tensor = None
        audio_tensor = None

        if image is not None:
            image_data = load_image(str(image))
            image_tensor = self.image_processor(image_data, return_tensors='pt')['pixel_values'].half().cuda()
            # just one turn, always prepend image token
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        
        if video is not None:
            video_tensor = self.video_processor(video, return_tensors='pt')['pixel_values'].half().cuda()
            inp = DEFAULT_VIDEO_TOKEN + '\n' + prompt
        
        if audio is not None:
            audio_tensor = self.audio_processor(audio, return_tensors='pt')['pixel_values'].half().cuda()
            inp = DEFAULT_AUDIO_TOKEN + '\n' + prompt
        # print("video_tensor: ", video_tensor.shape)
        # image_tensor = self.image_processor.preprocess(image_data, return_tensors='pt')['pixel_values']
    
    
        # just one turn, always prepend image token
        # inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        # inp = DEFAULT_VIDEO_TOKEN + '\n' + prompt
        # inp = prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        _prompt = conv.get_prompt()
        print("prompt: ", _prompt)
        input_ids = tokenizer_multiple_token(_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        print("input_ids: ", input_ids)
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
    
        
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                images=image_tensor,
                videos = video_tensor,
                audios = audio_tensor,
                **generation_args.__dict__
            )
            
            print("output: ", output)
            print("output shape: ", self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=False)[0])
            
            output_sequence = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=False)[0]
            
            final_output, module, instruction = self.parse_model_output(output_sequence)
            print("final_output: ", final_output)
            print("module: ", module)
            print("instruction: ", instruction)
            if module == 'ImageSegmentation':
                # image segmentation
                for inst in instruction:
                    image, mask, labels = self.image_segmentation(image, inst)
                    return image, mask, labels
            
            elif module == 'VideoTracking':
                # video tracking
                for inst in instruction:
                    video = self.video_tracking(video_path=image, sketch_pad=None, track_prompt=inst, text_prompt="")
                    return video
            elif module == 'AudioSegmentation':
                # audio segmentation
                for inst in instruction:
                    audio = self.audio_segmentation(audio_path=image, path_to_checkpoint="", path_to_segmentation_yaml="")
                    return audio
            else:
                return final_output
    

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    # show me a beautiful landscape of 
    # descibe the bird in the image
    # show me an image of a cute dog running on the grass
    predictor.predict(image="./assets/bird_image.jpg", prompt="descibe the image of the bird")
    # for i in predictor.predict(image="https://www.istockphoto.com/photo/abstract-background-of-geometric-figures-gm1282524189-380051534", prompt="A beautiful landscape with"):
    #     print(i)

