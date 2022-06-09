
import random
import torch
import numpy as np
import json
NUM_FOLD = 5
QUESTION_WORD_LIST = ["what", "when", "why", "who", "how", "where", "whose", "is", "are", "were", "was", "do", "does", "did", "can","could"]
OTHER_WORD_LIST = ["tell","please","request","allow","need","want","give","assign"]
special_tokens_dict = {'sep_token': '<SEP>', 'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>', 'cls_token': '<CLS>'}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def convert_json_to_txt(json_file, out_file, key='output'):
    print(f"converting {json_file} for {key}...")
    with open(json_file, encoding="utf-8") as fp, open(out_file,'w') as rp:
        for line in fp:
            data = json.loads(line)
            # print(data)
            if data['needs_rewrite']:
                rp.write(data[key])
                rp.write('\n')
    
    print("DONE!")