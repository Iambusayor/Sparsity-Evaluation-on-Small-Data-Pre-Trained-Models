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
"""
Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task.
"""
import logging
import os
import csv
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for token classification.
    """

    def __init__(self, guid, words, labels):
        """
        Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """
    A single set of features of data.
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if len(line) < 2 or line == "\n":
                print(line, words)
                if words:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words,
                            labels=labels,
                        )
                    )
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(
                InputExample(
                    guid="{}-{}".format(mode, guid_index), words=words, labels=labels
                )
            )
    return examples


def generate_misspelling(line, word_substitution_prob=0.2):
    keyboard_proximity = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdfr', 'f': 'drtgvc',
        'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'j': 'hnuikm', 'k': 'jiolm', 'l': 'kop',
        'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'azxdc',
        't': 'rfgy', 'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
    }

    def misspell_word(word):
        misspelled_word = list(word)
        for i in range(len(word)):
            if random.random() < word_substitution_prob:
                if word[i] in keyboard_proximity:
                    misspelled_word[i] = random.choice(keyboard_proximity[word[i]])
                else:
                    misspelled_word[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
        return ''.join(misspelled_word)

    words = line.split()  # Split the line into words
    misspelled_words = [
        misspell_word(word) if random.random() < word_substitution_prob else word for word in words
    ]
    return ' '.join(misspelled_words)


def read_examples_from_file_noise(data_dir, mode, word_substitution_prob=0.2):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            line = line.strip()
            if len(line) < 2 or line == "\n":
                if words:
                    # Call the generate_misspelling function to generate misspelled versions of the words
                    misspelled_words = [generate_misspelling(word, word_substitution_prob) for word in words]

                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=misspelled_words,
                            labels=labels,
                        )
                    )
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            # Call the generate_misspelling function for the last example
            misspelled_words = [generate_misspelling(word, word_substitution_prob) for word in words]

            examples.append(
                InputExample(
                    guid="{}-{}".format(mode, guid_index), words=misspelled_words, labels=labels
                )
            )
    return examples


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of `InputBatch`s `cls_token_at_end` define the location of the CLS
    token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for ex_index, example in enumerate(examples):
        # print(ex_index, len(example.words))
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend(
                [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)
            )

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        try:
            assert len(label_ids) == max_seq_length
        except:
            continue

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
            )
        )
    return features


def get_labels(path=None):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return [
            "O",
            "B-DATE",
            "I-DATE",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
        ]


class NERDataset(Dataset):
    def __init__(self, input_ids, input_labels, input_masks) -> None:
        self.examples = np.array(
            [
                {"input_ids": ids, "labels": labels, "attention_mask": masks}
                for ids, labels, masks in zip(input_ids, input_labels, input_masks)
            ]
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.examples[index]

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    return 'Size (MB):', os.path.getsize("temp.p")/1e6
    os.remove('temp.p')

def write_csv(results, lang_seed, op, file_path, model_size, model_parameters, inference_time,run_name):
    if not os.path.exists(file_path):
        # If file doesn't exist, create it with headers
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "loss",
                    "precision",
                    "recall",
                    "f1",
                    "eval/test",
                    "lang_seed",
                    "run_name",
                    "inference_time",
                    "model_size",
                    "model_parameters",
                ]
            )
            logger.info(f"{file_path} created with headers")

    results["eval/test"] = op
    results["lang_seed"] = lang_seed
    results["run_name"] = run_name
    results["inference_time"] = inference_time
    results["model_size"] = model_size
    results["model_parameters"] = model_parameters
    # Add rows to the file
    
    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(results.values())
        logger.info(f"Results logged to csv file at path {file_path}")
