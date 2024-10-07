# Copyright 2024 THU-BPM MarkLLM.
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

# Build a unified dictionary from external dictionaries

import random
import argparse
from tqdm import tqdm
from opencc import OpenCC
from itertools import permutations

# set seed
random.seed(0)

S2T = OpenCC('s2t')
T2S = OpenCC('t2s')

def transform(token, append_meta_symbols):
    def capitalize(s):
        return s.capitalize()

    def s2t(s):
        return S2T.convert(s)

    def t2s(s):
        return T2S.convert(s)

    def add_meta_symbols(s):
        # TODO: currently we only consider the sentencepiece meta symbol (U+2581)
        return f"▁{s}"

    # all permutations of the transformations
    transformations = []

    if append_meta_symbols:
        for r in range(1, 5):
            transformations.extend(permutations([capitalize, s2t, t2s, add_meta_symbols], r))
    else:
        for r in range(1, 4):
            transformations.extend(permutations([capitalize, s2t, t2s], r))

    res = [token]
    for t in transformations:
        new_token = token
        for f in t:
            new_token = f(new_token)
        res.append(new_token)

    # deduplicate
    res = list(set(res))
    return res

def augment_dictionary(raw_entries, append_meta_symbols):
    augmented_entries = []
    for src, tgt in tqdm(raw_entries, desc="Augmenting dictionary"):
        src_tokens = transform(src, append_meta_symbols)
        tgt_tokens = transform(tgt, append_meta_symbols)
        for src_token in src_tokens:
            for tgt_token in tgt_tokens:
                if src_token != tgt_token:
                    augmented_entries.append((src_token, tgt_token))

    # deduplicate
    augmented_entries = list(set(augmented_entries))
    return augmented_entries

def main(args):
    # Read data
    raw_entries = [] # list of tuples (src, tgt)
    for d_path in args.dicts:
        with open(d_path, "r") as f:
            for line in f:
                src, tgt = line.strip().split()
                if src != tgt:
                    raw_entries.append((src, tgt))

    # Augment dictionary
    augmented_entries = augment_dictionary(raw_entries, args.append_meta_symbols)

    # Write data
    with open(args.output_file, "w") as f:
        for src, tgt in augmented_entries:
            f.write(f"{src}\t{tgt}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a unified dictionary from external dictionaries")
    parser.add_argument("--dicts", type=str, nargs="+", help="multiple external dictionaries")
    parser.add_argument("--output_file", type=str, help="output dictionary")
    parser.add_argument("--append_meta_symbols", action="store_true", help="append meta symbols to the tokens")

    args = parser.parse_args()
    main(args)
