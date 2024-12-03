# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
import os 
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import Sampler, RandomSampler
from torch.utils.data import DataLoader, Dataset
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
import datasets
from typing import List, Optional

from reamo.dataset.sampler import DistributedMultiDatasetBatchSampler


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]
    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [(0) for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")
    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    """
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


class ReamoTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            # print(lengths)
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        elif self.args.group_by_modality_type:
            sampler = torch.utils.data.RandomSampler(self.train_dataset)
            rank = torch.distributed.get_rank()
            # print(self.train_dataset)
            # print(self.args.train_batch_size)  # 4 
            # print(rank)  # 0
            # print(self.args.world_size)  # 1
            # print(self.args.gradient_accumulation_steps)  # 1
            # print(sampler)
            # world_size = torch.distributed.get_world_size()
            # batch_size = self.args.world_size * self.args.config['train_micro_batch_size_per_gpu']
            # sampler = torch.utils.data.RandomSampler(concat_data)
            print("Building batch sampler 1111 ...")
            return DistributedMultiDatasetBatchSampler(batch_size=self.args.train_batch_size * self.args.world_size, 
                                                       sampler=sampler,
                                                       dataset=self.train_dataset,
                                                       rank=rank,
                                                       drop_last=True,
                                                       world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                                                       )
        else:
            return super()._get_train_sampler()

    # def get_train_dataloader(self) -> DataLoader:
    #     """
    #     Returns the training [`~torch.utils.data.DataLoader`].

    #     Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    #     training if necessary) otherwise.

    #     Subclass and override this method if you want to inject some custom behavior.
    #     """
        
    #     if self.train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")

    #     train_dataset = self.train_dataset
    #     data_collator = self.data_collator
    #     if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
    #         train_dataset = self._remove_unused_columns(train_dataset, description="training")
    #     else:
    #         data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    #     if self.args.group_by_modality_type:
    #         dataloader_params = {
    #             # "batch_size": self._train_batch_size,
    #             "collate_fn": data_collator,
    #             "num_workers": self.args.dataloader_num_workers,
    #             "pin_memory": self.args.dataloader_pin_memory,
    #             "persistent_workers": self.args.dataloader_persistent_workers,
    #         }
    #     else:
    #         dataloader_params = {
    #             "batch_size": self._train_batch_size,
    #             "collate_fn": data_collator,
    #             "num_workers": self.args.dataloader_num_workers,
    #             "pin_memory": self.args.dataloader_pin_memory,
    #             "persistent_workers": self.args.dataloader_persistent_workers,
    #         }
    #     if not isinstance(train_dataset, torch.utils.data.IterableDataset):
    #         if self.args.group_by_modality_type:
    #             print("Building batch sampler")
    #             dataloader_params["batch_sampler"] = self._get_train_sampler()
    #         else:
    #             dataloader_params["sampler"] = self._get_train_sampler()
    #             dataloader_params["drop_last"] = self.args.dataloader_drop_last
    #         dataloader_params["worker_init_fn"] = seed_worker
    #         dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
    #     print("data_loader_params: ", dataloader_params)
    #     return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def create_optimizer_groups(self, opt_model, decay_parameters, input_projector_parameters, output_projector_parameters, input_lr, output_lr):
        """Creates optimizer parameter groups without overlapping parameters."""
        grouped_params = {
            "weight_decay": [],
            "no_decay": [],
            "input_projector": [],
            "output_projector": []
        }

        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue

            # Determine if the parameter is a decay parameter
            is_decay = name in decay_parameters

            # Check for input and output projector membership
            is_input_projector = name in input_projector_parameters
            is_output_projector = name in output_projector_parameters

            # Assign parameters to groups avoiding overlaps
            if is_input_projector:
                grouped_params["input_projector"].append(param)
            elif is_output_projector:
                grouped_params["output_projector"].append(param)
            elif is_decay:
                grouped_params["weight_decay"].append(param)
            else:
                grouped_params["no_decay"].append(param)

        # Create the actual optimizer groups
        optimizer_groups = [
            {"params": grouped_params["weight_decay"], "weight_decay": self.args.weight_decay},
            {"params": grouped_params["no_decay"], "weight_decay": 0.0}
        ]

        if input_projector_parameters:
            optimizer_groups.append({"params": grouped_params["input_projector"], "lr": input_lr, "weight_decay": self.args.weight_decay})
        
        if output_projector_parameters:
            optimizer_groups.append({"params": grouped_params["output_projector"], "lr": output_lr, "weight_decay": self.args.weight_decay})

        return optimizer_groups

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_input_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_input_projector" in name]
                if self.args.mm_output_projector_lr is not None:
                    projector_parameters.extend([name for name, _ in opt_model.named_parameters() if ("mm_output_img_projector" in name or "mm_output_vid_projector" in name or "mm_output_aud_projector" in name)])
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_input_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_input_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        save_flag = False

        def save_adapter(keys_to_match, filename):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank in [0, -1]:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, filename))
            return True
        if getattr(self.args, 'tune_mm_input_adapter', False):
            print("Saving adapter in llava trainer in llava_trainer _save_checkpoint tune_mm_input_adapter ...")
            print(getattr(self.args, 'tune_mm_input_adapter', False))
            keys_to_match = ['mm_input_projector', 'vision_resampler', 'embed_tokens', 'embed_in']
            save_flag = save_adapter(keys_to_match, 'mm_input_projector.bin')

        # if any(getattr(self.args, f'tune_mm_output_{mod}_adapter', False) for mod in ['img', 'vid', 'aud']):
        #     print("Saving adapter in llava trainer in llava_trainer _save_checkpoint tune_mm_output_ ...")
        #     print(any(getattr(self.args, f'tune_mm_output_{mod}_adapter', False) for mod in ['img', 'vid', 'aud']))
        #     keys_to_match = ['mm_output_img_projector', 'mm_output_vid_projector', 'mm_output_aud_projector', 'embed_tokens', 'embed_in']
        #     save_flag = save_adapter(keys_to_match, 'mm_output_projector.bin')

        if not save_flag:
            super(ReamoTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # if getattr(self.args, 'tune_mm_input_adapter', False) or any(getattr(self.args, f'tune_mm_output_{mod}_adapter', False) for mod in ['img', 'vid', 'aud']):
        print("Saving model in llava trainer, _save ...")
        # print(getattr(self.args, 'tune_mm_input_adapter', False))
        if getattr(self.args, 'tune_mm_input_adapter', False):
            super(ReamoTrainer, self)._save(output_dir, state_dict)
        else:
            super(ReamoTrainer, self)._save(output_dir, state_dict)