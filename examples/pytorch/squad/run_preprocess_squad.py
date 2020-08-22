# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

import torch
import argparse
import logging
import tempfile
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    squad_convert_examples_to_features,
)
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor

logger = logging.getLogger(__name__)

from matorage import *

train_attributes = [
    ("all_input_ids", "int64", (384,)),
    ("all_attention_masks", "int64", (384,)),
    ("all_token_type_ids", "int64", (384,)),
    ("all_start_positions", "int64", (1,)),
    ("all_end_positions", "int64", (1,)),
    ("all_cls_index", "float32", (1,)),
    ("all_p_mask", "float32", (384,)),
    ("all_is_impossible", "float32", (1,)),
]

eval_attributes = [
    ("all_input_ids", "int64", (384,)),
    ("all_attention_masks", "int64", (384,)),
    ("all_token_type_ids", "int64", (384,)),
    ("all_feature_index", "int64", (1,)),
    ("all_cls_index", "int64", (1,)),
    ("all_p_mask", "float32", (384,)),
]

def data_save(args, tokenizer, evaluate=False):

    data_config = DataConfig(
        endpoint="127.0.0.1:9000",
        access_key="minio",
        secret_key="miniosecretkey",
        dataset_name="SQuAD1.1" if not args.version_2_with_negative else "SQuAD2.0",
        additional={
            "mode": "train" if not evaluate else "test",
            "framework": "pytorch",
            "version": 1.1 if not args.version_2_with_negative else 2.0,
            "model_name": args.tokenizer_name,
            "doc_stride": args.doc_stride,
            "max_seq_length": args.max_seq_length,
            "max_query_length": args.max_query_length,
        },
        attributes=train_attributes if not evaluate else eval_attributes,
    )

    processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
    if evaluate:
        examples = processor.get_dev_examples(None, filename=args.predict_file)
    else:
        examples = processor.get_train_examples(None, filename=args.train_file)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=args.threads,
    )

    data_saver = DataSaver(config=data_config)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=args.threads)

    if not evaluate:
        for batch in tqdm(dataloader):
            inputs = {
                "all_input_ids": batch[0],
                "all_attention_masks": batch[1],
                "all_token_type_ids": batch[2],
                "all_start_positions": batch[3],
                "all_end_positions": batch[4],
                "all_cls_index": batch[5],
                "all_p_mask": batch[6],
                "all_is_impossible": batch[7]
            }
            data_saver(inputs)
    else:
        for batch in tqdm(dataloader):
            inputs = {
                "all_input_ids": batch[0],
                "all_attention_masks": batch[1],
                "all_token_type_ids": batch[2],
                "all_feature_index" : batch[3],
                "all_cls_index": batch[4],
                "all_p_mask": batch[5],
            }
            data_saver(inputs)
        _features, _examples  = tempfile.mktemp("features"), tempfile.mktemp("examples")
        torch.save(features, _features)
        torch.save(examples, _examples)
        data_saver({"features" : _features, "examples" : _examples}, filetype=True)

    data_saver.disconnect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        required=True,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        default=True,
        help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.do_train:
        data_save(args, tokenizer, evaluate=False)
    if args.do_eval:
        data_save(args, tokenizer, evaluate=True)


if __name__ == "__main__":
    main()
