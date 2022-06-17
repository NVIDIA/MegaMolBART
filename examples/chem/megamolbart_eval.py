# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


from argparse import ArgumentParser

import torch
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader

from nemo.collections.nlp.data.language_modeling.megatron.request_dataset import T5RequestDataset
from nemo_chem.models.megamolbart import MegaMolBARTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin, NLPSaveRestoreConnector
from nemo.utils.app_state import AppState

assert torch.cuda.is_available()

from torch.utils.data.dataset import Dataset

class MoleculeRequestDataset(Dataset):
    def __init__(self, request: Dict, tokenizer) -> None:
        super().__init__()
        self.request = request
        self.tokenizer = tokenizer

        # tokenize prompt
        self.request['tokenized_prompt'] = ' '.join(self.tokenizer.text_to_tokens(request['prompt']))
        tokens = self.tokenizer.text_to_ids(request['prompt'])
        self.request['tokens'] = torch.tensor(tokens)
        self.mask_prompt(self.request['prompt'])

    def mask_prompt(self, sample):
        sample = torch.LongTensor(self.tokenizer.text_to_ids(sample))
        self.request['masked_sample'] = sample

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.request


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str, default="/result/nemo_experiments/megamolbart/MegaMolBART-megamolbart_pretrain_small_span_aug.nemo", required=False, help="Pass path to model's .nemo file")
    parser.add_argument(
        "--prompt", type=str, default="N[C@H]1CCC(=O)[C@H](O)[C@H](O)[C@H]1O", required=False, help="Prompt for the model (a text to complete)"
    )
    parser.add_argument(
        "--tokens_to_generate", type=int, default="16", required=False, help="How many tokens to add to prompt"
    )
    parser.add_argument(
        "--tensor_model_parallel_size", type=int, default=1, required=False,
    )
    parser.add_argument(
        "--pipeline_model_parallel_size", type=int, default=1, required=False,
    )
    parser.add_argument(
        "--pipeline_model_parallel_split_rank", type=int, default=0, required=False,
    )
    parser.add_argument("--precision", default="16", type=str, help="PyTorch Lightning Trainer precision flag")
    args = parser.parse_args()

    # cast precision to int if 32 or 16
    if args.precision in ["32", "16"]:
        args.precision = int(float(args.precision))

    # trainer required for restoring model parallel models
    trainer = Trainer(
        plugins=NLPDDPPlugin(),
        devices=args.tensor_model_parallel_size * args.pipeline_model_parallel_size,
        accelerator='gpu',
        precision=args.precision,
    )

    app_state = AppState()
    if args.tensor_model_parallel_size > 1 or args.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            app_state.data_parallel_size,
            app_state.pipeline_model_parallel_split_rank,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=args.tensor_model_parallel_size,
            pipeline_model_parallel_size_=args.pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank_=args.pipeline_model_parallel_split_rank,
        )

    model = MoleculeRequestDataset.restore_from(
        restore_path=args.model_file, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector(),
    )
    model.freeze()

    request = {
        "prompt": args.prompt,
        "tokens_to_generate": args.tokens_to_generate,
    }

    dataset = T5RequestDataset(request, model.tokenizer)

    request_dl = DataLoader(dataset)

    response = trainer.predict(model, request_dl)

    print("***************************")
    print(response)
    print("***************************")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
