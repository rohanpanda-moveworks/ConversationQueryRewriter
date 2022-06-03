
import argparse
from faulthandler import disable
import logging
import json
import shutil
import collections
import collections.abc
import random
import os
from cqr.inference_model import InferenceModel
import torch
import torch.nn.functional as F
from datetime import datetime

from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm, trange
from transformers import  GPT2Config,GPT2DoubleHeadsModel,\
     GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

from cqr.dataset import QueryRewriteDataset
from cqr.utils import NUM_FOLD, set_seed, special_tokens_dict

logger = logging.getLogger(__name__)


def collate_fn(batch_dataset: list):
    return_tuple = [[], [], [], [], [], []]
    for example in batch_dataset:
        return_tuple[0].append(example.topic_number)
        return_tuple[1].append(example.query_number)
        return_tuple[2].append(example.ids)
        return_tuple[3].append(example.labels)
        return_tuple[4].append(example.pred_begin_pos)
        return_tuple[5].append(example.needs_rewrite)
    return_tuple[2] = torch.tensor(return_tuple[2])
    return_tuple[3] = torch.tensor(return_tuple[3])
    return_tuple = tuple(return_tuple)
    return return_tuple

def get_lm_loss(preds, target, needs_rewrite):
    # print(preds.shape, target.shape)
    needs_rewrite = needs_rewrite.repeat(1,preds.shape[1]).view(-1)
    preds = preds.view(-1,preds.shape[-1])
    target = target.view(-1)
    
    # print(target)
    loss = F.cross_entropy(preds, target, reduction='none', ignore_index=-1)
    
    loss = (loss.view(-1)*needs_rewrite).sum()
    if torch.isnan(loss):
        print("called from loss calc")
        print(preds, target, needs_rewrite)
        exit(0)
    nr = needs_rewrite.sum().item()
    nr = nr if nr > 0 else 1
    return loss/nr

def eval(args, val_dataset, model, inf_model, tokenizer , logger):
    args.val_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    val_sampler = RandomSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)
    model.eval()
    inf_model.model = model  # updating model before decoding

    if args.max_steps > 0:
        t_total = args.max_steps
        num_val_epochs = args.max_steps // (len(val_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(val_dataloader) 

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    # val_iterator = trange(int(num_val_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    epoch_iterator = tqdm(val_dataloader, desc="Iteration", \
        disable=args.local_rank not in [-1, 0])
    epoch_pos, epoch_tot = 0., 0.
    for step, batch in enumerate(epoch_iterator):
        inputs, labels = (batch[2], batch[3])  # get ids and labels
        inputs = inputs.to(args.device)  # batch_size * block_size
        labels = labels.to(args.device)
        mc_labels = torch.tensor(batch[5]).to(args.device)
        mc_token_ids = (inputs == tokenizer.cls_token_id).nonzero(as_tuple=False)
        mc_token_ids = mc_token_ids[:,1]
        mc_token_ids = mc_token_ids.to(args.device)
        model.eval()
        outputs = model(input_ids=inputs, lm_labels=labels, mc_labels=mc_labels, mc_token_ids=mc_token_ids)
        mc_loss = outputs[1]  # model outputs are always tuple in transformers (see doc)
        lm_loss = get_lm_loss(outputs[2],labels,mc_labels)
        loss = mc_loss + lm_loss
        epoch_tot += len(labels)
        _,pred = outputs[3].data.topk(1,dim=1)
        pred = pred.flatten()
        # print(pred.shape,mc_labels.shape)
        epoch_pos += (pred == mc_labels).sum().item()
        
        del inputs
        del outputs
        torch.cuda.empty_cache()

        if args.n_gpu > 1:
            loss = loss.sum()  # mean() to average on multi-gpu parallel training
        # if args.gradient_accumulation_steps > 1:
        #     loss = loss / args.gradient_accumulation_steps

        tr_loss += loss.item()
        global_step += 1
        
        torch.cuda.empty_cache()
        frac = ((step+1)/len(epoch_iterator))
        epoch_iterator.set_postfix(Loss=tr_loss/global_step, Acc=epoch_pos/epoch_tot*100)

        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
        del loss
    
    if args.debug:
        with open(args.valid_file, 'r') as valid, open(args.train_file, 'r') as train:
            random_val, random_train = random.choice(valid.readlines()), random.choice(train.readlines())
            val_pt, train_pt = json.loads(random_val), json.loads(random_train)
            # print(val_pt,train_pt)
            val_pred = inf_model.predict(val_pt['input'])
            train_pred = inf_model.predict(train_pt['input'])
            logger.info("************DEBUG*********")
            logger.info(f"Train Target: {train_pt['target']} \n Train pred: {train_pred}")
            logger.info(f"Val Target: {val_pt['target']} \n Val pred: {val_pred}")
    return tr_loss / global_step, epoch_pos/epoch_tot


def train(args, train_dataset, val_dataset, model, inf_model, tokenizer, logger, cross_validate_id=-1):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(f" Num GPU = {args.n_gpu}")
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # eval(args, val_dataset, model, inf_model, tokenizer, logger)
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch",\
        disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for ep in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", \
            disable=args.local_rank not in [-1, 0])
        epoch_loss = 0.
        epoch_pos, epoch_tot = 0., 0.
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = (batch[2], batch[3])  # get ids and labels
            # print(inputs, tokenizer.cls_token_id)
            inputs = inputs.to(args.device)  # batch_size * block_size
            labels = labels.to(args.device)
            mc_labels = torch.tensor(batch[5]).to(args.device)
            # mc_token_ids = torch.tensor([inputs.index(tokenizer.cls_token_id)\
            #      for _ in inputs]).to(args.device)
            mc_token_ids = (inputs == tokenizer.cls_token_id).nonzero(as_tuple=False)
            # print(mc_token_ids, mc_token_ids.shape)
            mc_token_ids = mc_token_ids[:,1]
            # print(mc_token_ids)
            # print(mc_token_ids.shape, inputs.shape)
            # exit(0)
            mc_token_ids = mc_token_ids.to(args.device)
            # print(inputs.shape, labels.shape, mc_labels.shape, mc_token_ids.shape)
            # print(inputs[12])
            model.train()
            outputs = model(input_ids=inputs, lm_labels=labels, mc_labels=mc_labels, mc_token_ids=mc_token_ids)
            mc_loss = outputs[1]  # model outputs are always tuple in transformers (see doc)
            lm_loss = get_lm_loss(outputs[2],labels,mc_labels)
            if torch.isnan(lm_loss):
                print(f"called during train, cls is {tokenizer.cls_token_id}")
                print(f"Input: {inputs}\n \
                        label: {labels}\n \
                        mc_loss: {mc_loss}\n \
                        lm_loss: {lm_loss}")
                exit(0)
            loss = mc_loss + 10*lm_loss
            epoch_tot += len(labels)
            _,pred = outputs[3].data.topk(1,dim=1)
            pred = pred.flatten()
            # print(pred,mc_labels)
            # exit(0)
            epoch_pos += (pred == mc_labels).sum().item()
            
            del inputs
            del outputs
            torch.cuda.empty_cache()

            if args.n_gpu > 1:
                loss = loss.sum()  # mean() to average on multi-gpu parallel training
            # if args.gradient_accumulation_steps > 1:
            #     loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            epoch_loss += loss.item()
            # print(epoch_loss/(step+1))
            # exit(0)
            del loss
            torch.cuda.empty_cache()
            frac = ((step+1)/len(epoch_iterator))
            epoch_iterator.set_postfix(Loss=epoch_loss/(step+1),Pos=epoch_pos ,Acc=epoch_pos/epoch_tot*100)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    output_dir = args.output_dir + (('-' + str(cross_validate_id)) if cross_validate_id != -1 else "")
                    # Save model checkpoint
                    output_dir = os.path.join(output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        
        epoch_loss /= len(epoch_iterator)
        epoch_acc = epoch_pos/epoch_tot*100
        logger.info(f"==========Epoch {ep}/{int(args.num_train_epochs)}==========")
        logger.info(f"Train Loss: {epoch_loss} | Train Acc: {epoch_acc}")
        val_loss, val_acc = eval(args, val_dataset, model, inf_model, tokenizer, logger)
        logger.info(f"Val Loss: {val_loss} | Train Acc: {val_acc}")
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_name_or_path", default="gpt2-medium", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--block_size", default=200, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="Path of training file. Do not add fold suffix when cross validate, i.e. use 'data/eval_topics.jsonl' instead of 'data/eval_topics.jsonl.0'")
    parser.add_argument("--valid_file", default=None, type=str, required=True,
                        help="Path to validation file.")
    parser.add_argument("--cross_validate", action='store_true',
                        help="Set when doing cross validation")
    parser.add_argument("--init_from_multiple_models", action='store_true',
                        help="Set when initialize from different models during cross validation (Model-based+CV)")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--n_gpu', type=int, default=2,
                        help="number of GPUs to use")
    parser.add_argument('--mtl', action='store_true',\
                        help="Use this flag for Multi-task learning")
    parser.add_argument('--debug', action="store_true",
                        help="Enable to print one decoded example during training")
    parser.add_argument("--length", type=int, default=20,
                        help="Maximum length of output sequence")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--toy_data", action="store_true",
                        help="use only 100 datapoints for debugging")
    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count() if args.n_gpu < 1 else args.n_gpu

    if args.overwrite_output_dir:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    # Setup logging
    os.makedirs(args.output_dir,exist_ok=True)
    log_file_path = os.path.join(args.output_dir,datetime.now().strftime('MTL_%H_%M_%d_%m_%Y.log'))
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                        filename=log_file_path)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    config_class, model_class, tokenizer_class = GPT2Config, GPT2DoubleHeadsModel, GPT2Tokenizer

    if not args.cross_validate:
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens(special_tokens_dict)
        model = model_class.from_pretrained(args.model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))  # resize
        model.to(args.device)
        model_config = {'model': model, 'tokenizer': tokenizer}
        inf_model = InferenceModel(args, model_config)
	
        if args.block_size <= 0:
            args.block_size = tokenizer.max_len_single_sentence
        args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

        # Training
        logger.info("Training/evaluation parameters %s", args)
        train_dataset = QueryRewriteDataset([args.train_file], tokenizer, args, debugging=args.toy_data)
        val_dataset = QueryRewriteDataset([args.valid_file], tokenizer, args, debugging=args.toy_data)
        global_step, tr_loss = train(args, train_dataset, val_dataset, model, inf_model, tokenizer, logger)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    else:
        # K-Fold Cross Validation
        for i in range(NUM_FOLD):
            logger.info("Training Fold #{}".format(i))
            suffix = ('-' + str(i)) if args.init_from_multiple_models else ''
            config = config_class.from_pretrained(args.model_name_or_path + suffix)
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path + suffix)
            tokenizer.add_special_tokens(special_tokens_dict)
            model = model_class.from_pretrained(args.model_name_or_path + suffix)
            model.resize_token_embeddings(len(tokenizer))  # resize
            model.to(args.device)

            if args.block_size <= 0:
                args.block_size = tokenizer.max_len_single_sentence
            args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    
            logger.info("Training/evaluation parameters %s", args)
            train_files = ["%s.%d" % (args.train_file, j) for j in range(NUM_FOLD) if j != i]
            logger.info("train_files: {}".format(train_files))
            train_dataset = QueryRewriteDataset(train_files, tokenizer, args)
            global_step, tr_loss = train(args, train_dataset, model, inf_model, tokenizer, logger, cross_validate_id=i)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

            # Create output directory if needed
            output_dir = args.output_dir + '-' + str(i)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            logger.info("Saving model checkpoint to %s", output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))

            del model
            torch.cuda.empty_cache() 


if __name__ == "__main__":
    main()

