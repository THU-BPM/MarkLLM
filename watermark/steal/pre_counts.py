import argparse
from countstore import CountStore
from transformers import AutoTokenizer
import json
import os
from tqdm import tqdm


def format_str_list(standard_str:str, rp_keys, rp_sign="[R]"):
    res = []
    for key in rp_keys:
        key_str = standard_str.replace(rp_sign, key)
        res.append(key_str)
    return res


def format_integer(to_format, format_count):
    str_a = str(to_format)
    leading_zeros = format_count - len(str_a)
    
    if leading_zeros > 0:
        return '0' * leading_zeros + str_a
    else:
        return str_a[:format_count]


def return_files_list(input:str):
    ''' get files in input dir, must have suffix .jsonl'''
    if os.path.isfile(input):
        return [input]
    else:
        # init to save files
        file_paths = []
        # os.walk to get all content
        for root, dirs, files in os.walk(input):
            for file in files:
                # get absolute path
                file_path = os.path.abspath(os.path.join(root, file))
                # config is json file with suffix '.json', data file with suffix .jsonl
                if file_path.endswith(".jsonl"):
                    file_paths.append(file_path)
        return file_paths


def create_counts_file(model_name_or_path:str, files: list, steal_key:str="text", prevctx_width: int = 1, record_count: int=5000, add_flag: bool=False, save_name: str="tem_counts.pkl", old_counts: CountStore=None):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # get counts
    if add_flag:
        counts = old_counts
    else:
        counts = CountStore(prevctx_width=prevctx_width)
    
    # load steal_file
    count = 0
    for key_id, file in enumerate(files):
        for id, x in tqdm(enumerate(open(file, encoding="utf-8")), desc=f"FILE{(key_id+1)}"):
            # load data and get text
            example = json.loads(x)
            text = example[steal_key]
            toks = tokenizer(text)["input_ids"]
            if isinstance(toks[0], list):
                for b in range(len(toks)):
                    for i in range(prevctx_width, len(toks[b])):
                        ctx = tuple(toks[b][i - prevctx_width : i])
                        counts.add(ctx, toks[b][i], 1)
            else:
                for i in range(prevctx_width, len(toks)):
                    ctx = tuple(toks[i - prevctx_width : i])
                    counts.add(ctx, toks[i], 1)
            count += 1
            if count >= record_count:
                break
        if count >= record_count:
            break
    
    counts.save_file(save_name)
    print(counts.nb_keys(True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="for watermark and detect")
    
    # =============model
    parser.add_argument("--model", default="opt-2.7b", type=str, help="the model for spoofing attack")
    
    # ============counts
    parser.add_argument("--prevctx_width", default=None, type=int, help="the prefix context width")
    parser.add_argument("--old_counts", default=None, type=str, help="an exist counts file, 使用新文件")
        
    
    # =============datas
    parser.add_argument("--steal_file", default=None, type=str, help="a dir or a file, all file with suffix .jsonl")
    parser.add_argument("--steal_key", default="text", type=str, help="text key")
    parser.add_argument("--record_count", default=5000, type=int, help="steal complex")
    parser.add_argument("--save_file", default=None, type=str, help="save new counts file")
    args = parser.parse_args()
    
    # get files
    if os.path.isdir(str(args.steal_file)):
        files = return_files_list(args.steal_file)
    else:
        files = [str(args.steal_file)]
    
    create_paras = {
        "model_name_or_path":args.model,
        "files":files,
        "steal_key":args.steal_key,
        "prevctx_width":int(args.prevctx_width),
        "record_count": int(args.record_count),
        "save_name": str(args.save_file)
    }
    
    if args.old_counts is None:
        create_counts_file(**create_paras)
    else:
        old_counts = CountStore(prevctx_width=int(args.prevctx_width))
        old_counts.init_from_file(args.old_counts)
        create_counts_file(add_flag=True, old_counts=old_counts, **create_paras)
    