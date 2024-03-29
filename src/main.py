#!/usr/bin/env python
__author__ = "Ricardo Jung"
__email__ = "s2458588@stud.uni-frankfurt.de"

# __copyright__ = ""
# __credits__ = [""]
# __license__ = ""
# __version__ = ""
# __maintainer__ = ""
# __status__ = ""

import datasets
import wm_tokenizer
import text_utilities as tu
from HanTa import HanoverTagger as ht
from transformers import BertTokenizer, Trainer, TrainingArguments, AutoModelForMaskedLM # AutoModelForSequenceClassification
from tokenizers import pre_tokenizers
import torch

vd = tu.VerbDict("../new_tokenizer/fun_vocab_raw.txt", "../new_tokenizer/lex_vocab_raw.txt")

wmt = wm_tokenizer.WordmapTokenizer(
    bert_pretokenizer=pre_tokenizers.BertPreTokenizer(),
    bert_tokenizer=BertTokenizer.from_pretrained("bert-base-german-cased"),
    hantatagger=ht.HanoverTagger('morphmodel_ger.pgz'),
    vocab=vd.lmfm
)


def wm_tokenize(data):
    return wmt.wordmap2tokenizer(data['text'], pos_tag="V", vocab=wmt.vocab, pt=wmt.bert_pretokenizer,
                                 tk=wmt.bert_tokenizer, tg=wmt.hantatagger)


def main():

    files = tu.files_from_path(sys.argv[2], full_path=True)
    dataset = datasets.load_dataset("text", data_files=files, split="train")
    dataset = dataset.train_test_split(train_size=1000, test_size=150, writer_batch_size=1000)
    metric = datasets.load_metric('glue', 'mrpc', keep_in_memory=True)

    tokenized_dataset = dataset.map(wm_tokenize, batched=True, batch_size=1000)

    # recommendations: https://github.com/google-research/bert
    training_args = TrainingArguments(
        output_dir='./out/model_out',  # output directory
        num_train_epochs=4,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./out/model_logs',  # directory for storing logs
        logging_steps=10,
        learning_rate=3e-4
    )

    model = AutoModelForMaskedLM.from_pretrained(pretrained)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],  # training dataset
        eval_dataset=tokenized_dataset["test"]  # evaluation dataset
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = trainer.model.to(device)
    model.save_pretrained("../out/model/model_out_mlm.bin")
    print(model, "trained on", data_path, "saved in", output_path)

if __name__ == '__main__':
    import sys

    pretrained = sys.argv[1]  # "bert-base-german-cased"
    data_path = sys.argv[2]  # "../data/oscar/to_lines"
    output_path = sys.argv[3] # "../out/model/"
