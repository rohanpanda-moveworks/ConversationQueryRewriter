import json
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', 
                        type=str, 
                        required=True, 
                        help="inpt json file path for CANARD data"
                        )
    parser.add_argument('--output_path', 
                        type=str, 
                        required=True, 
                        help="outpt json file path for inverted CANARD data"
                        )
    parser.add_argument('--simplify',
                        action='store_true',
                        help="whether canard needs to be used for simplifier or  org training"
                        )
    args = parser.parse_args()

    canard_data = []
    with open(args.input_path, 'r') as f:
        canard_data = json.loads(f.read())

    invert_canard_data = []
    curr_id = "null"
    topic_num = 1
    for inst in canard_data:
        pt = dict()
        if inst['QuAC_dialog_id'] != curr_id:
            topic_num += 1
            curr_id = inst['QuAC_dialog_id']
        
        pt['topic_number'] = topic_num
        pt['query_number'] = inst['Question_no'] + 1 
        if args.simplify:
            tgt, hist = inst["Question"], inst["Rewrite"]
        else:
            tgt, hist = inst["Rewrite"], inst["Question"]

        pt['input'] = inst["History"] + [hist]
        pt['target'] = tgt
        invert_canard_data.append(pt)

    with open(args.output_path, 'w') as f:
        for line in invert_canard_data:
            f.write(json.dumps(line)+ '\n')

if __name__ == '__main__':
    main()


