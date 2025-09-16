import os
import nltk
import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm
from comprehensive_eval import comprehensive_evaluation, save_comprehensive_results
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
# from transformers import BertTokenizer, EncoderDecoderModel
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from transformers import BartTokenizerFast

from data_utils import ABSADataset
from data_utils import read_line_examples_from_file



logger = logging.getLogger(__name__)


def debug_log(msg):
    with open('src/debug/debug.log', 'a', encoding='utf-8') as f:
        f.write(msg + '\\n')
    print(msg)

def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='asqp', type=str, required=True,
                        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--dataset", default='sim7', type=str, required=True,
                        help="The name of the dataset, selected from: [sim7,dim7]")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true', 
                        help="Whether to run inference with trained checkpoints")
    parser.add_argument("--backbone", default='t5', type=str, choices=['t5', 't5-large', 'bart', 'bart-large'],
        help="choose backbone: t5、t5-large、bart 或 bart-large")

    # 新增参数
    parser.add_argument("--use_instruction", action='store_true',
                        help="Whether to use instruction prompt in input")
    parser.add_argument("--instruction_template", default="Analyze sentiment: ",
                        help="Instruction template to prepend to input")

    # other parameters
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--n_gpu", default=0, type=int,
                        help="GPU device id, use 0 for first GPU")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=2025,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/t5/' or './outputs/rest15/bart/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')
    if args.use_instruction:
        output_dir = f"outputs/{args.dataset}/{args.backbone}_instruction"
    else:
        output_dir = f"outputs/{args.dataset}/{args.backbone}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.output_dir = output_dir

    return args



def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, 
                       data_type=type_path, max_len=args.max_seq_length, include_overall=True,
                       use_instruction=args.use_instruction, instruction_template=args.instruction_template)

class Seq2SeqFineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained sequence-to-sequence model (T5, BART, etc.)
    """
    def __init__(self, hparams, tfm_model, tokenizer):
        super(Seq2SeqFineTuner, self).__init__()
        self.hparams = hparams
        self.model = tfm_model
        self.tokenizer = tokenizer

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def evaluate(data_loader, model, sents, args):
    device = torch.device(f'cuda:{args.n_gpu}')
    model.model.to(device)
    model.model.eval()
    outputs, targets = [], []
    for batch in tqdm(data_loader):
        outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
                                    attention_mask=batch['source_mask'].to(device), 
                                    max_length=1024)
        dec = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
        outputs.extend(dec)
        targets.extend(target)
        
        print("Sample predictions from this batch:")
        for i, pred in enumerate(dec[:3]):   
            print(f"Prediction {i+1}: {pred}")

    results = comprehensive_evaluation(outputs, targets, sents)
    return results


# initialization
args = init_args()
if args.backbone == 't5':
    args.max_seq_length = 1024
elif args.backbone == 'bart':
    args.max_seq_length = 1024
elif args.backbone == 't5-large':
    args.max_seq_length = 1024
elif args.backbone == 'bart-large':
    args.max_seq_length = 1024


if hasattr(args, 'backbone'):
    if args.backbone == 't5':
        model_name_or_path = 'PLMS/t5-base'
    elif args.backbone == 't5-large':
        model_name_or_path = 'PLMS/t5-large'
    elif args.backbone == 'bart':
        model_name_or_path = 'PLMS/facebook-bart-base'
    elif args.backbone == 'bart-large':
        model_name_or_path = 'PLMS/bart-large'
    else:
        raise ValueError('Unsupported backbone')
else:
    model_name_or_path = 'PLMS/t5-base'

print("\n", "="*30, f"NEW EXP: ASQP on {args.dataset} (backbone: {args.backbone})", "="*30, "\n")


if args.backbone in ['t5', 't5-large']:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
elif args.backbone in ['bart', 'bart-large']:
    tokenizer = BartTokenizerFast.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"设置BART pad_token为: {tokenizer.pad_token}")
else:
    raise ValueError('Unsupported backbone')

print(f"Here is an example (from the dev set):")
dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, 
                      data_type='dev', max_len=args.max_seq_length, include_overall=True,
                      use_instruction=args.use_instruction, instruction_template=args.instruction_template)
data_sample = dataset[7]  # a random data sample
print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))

# training process
if args.do_train:
    print("\n****** Conduct Training ******")
    if args.backbone in ['t5', 't5-large']:
        from transformers import T5ForConditionalGeneration
        tfm_model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    elif args.backbone in ['bart', 'bart-large']:
        from transformers import BartForConditionalGeneration
        tfm_model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
    else:
        raise ValueError('Unsupported backbone')
    model = Seq2SeqFineTuner(args, tfm_model, tokenizer)
    # prepare for trainer
    train_params = dict(
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=[args.n_gpu] if args.n_gpu >= 0 else None,
        gradient_clip_val=1.0,
        max_epochs=args.num_train_epochs,
        callbacks=[LoggingCallback()],
        early_stop_callback=pl.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min'),
    )
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    # save the final model
    model.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Finish training and saving the model!")

# evaluation
def load_model_and_tokenizer_for_eval():
    if args.backbone in ['t5', 't5-large']:
        from transformers import T5ForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)
    elif args.backbone in ['bart', 'bart-large']:
        from transformers import BartForConditionalGeneration  # 必须加这一行
        tokenizer = BartTokenizerFast.from_pretrained(args.output_dir)
        # 修复BART的pad_token问题
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"评估时设置BART pad_token为: {tokenizer.pad_token}")
        tfm_model = BartForConditionalGeneration.from_pretrained(args.output_dir)
    else:
        raise ValueError('Unsupported backbone')
    return tfm_model, tokenizer

if args.do_eval:
    print("\n****** Conduct Evaluating with the last state ******")
    print(f"Load trained model from {args.output_dir}")
    tfm_model, tokenizer = load_model_and_tokenizer_for_eval()
    model = Seq2SeqFineTuner(args, tfm_model, tokenizer)  # 新增这一行
    sents, _, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt', silence=False)
    print()
    test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, 
                            data_type='test', max_len=args.max_seq_length, include_overall=True)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    results = evaluate(test_loader, model, sents, args)
    if args.use_instruction:
        log_file_path = f"results_log/{args.dataset}_{args.backbone}_instruction.txt"
    else:
        log_file_path = f"results_log/{args.dataset}_{args.backbone}.txt"
    save_comprehensive_results(
        results,
        output_path=log_file_path,
        model_name=args.backbone,
        dataset_name=args.dataset
    )

if args.do_inference:
    print("\n****** Conduct inference on trained checkpoint ******")
    print(f"Load trained model from {args.output_dir}")
    print('Note that a pretrained model is required and `do_true` should be False')
    tfm_model, tokenizer = load_model_and_tokenizer_for_eval()
    model = Seq2SeqFineTuner(args, tfm_model, tokenizer)  # 新增这一行
    sents, _, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt', silence=False)
    print()
    test_dataset = ABSADataset(tokenizer, data_dir=args.dataset, 
                            data_type='test', max_len=args.max_seq_length, include_overall=True)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    results = evaluate(test_loader, model, sents, args)
    if args.use_instruction:
        log_file_path = f"results_log/{args.dataset}_{args.backbone}_instruction.txt"
    else:
        log_file_path = f"results_log/{args.dataset}_{args.backbone}.txt"
    save_comprehensive_results(
        results,
        output_path=log_file_path,
        model_name=args.backbone,
        dataset_name=args.dataset
    )