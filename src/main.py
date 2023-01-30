import datasets
import wm_tokenizer
import text_utilities
from HanTa import HanoverTagger as ht
from transformers import BertTokenizer, Trainer, TrainingArguments, BertForMaskedLM, AutoModelForMaskedLM
from tokenizers import pre_tokenizers

tu = text_utilities.VerbDict("../new_tokenizer/fun_vocab_raw.txt", "../new_tokenizer/lex_vocab_raw.txt")

wmt = wm_tokenizer.WordmapTokenizer(
    bert_pretokenizer=pre_tokenizers.BertPreTokenizer(),
    bert_tokenizer=BertTokenizer.from_pretrained("bert-base-german-cased"),
    hantatagger=ht.HanoverTagger('morphmodel_ger.pgz'),
    vocab=tu
)


def main():
    """TODO: Fix UNK tokens bei wmt.SequenceTokenizer"""

    dataset = datasets.Dataset = ...
    dataset = dataset.map(
        wmt.wordmap2tokenizer(sentence,
                              pos_tag="V",
                              vocab=wmt.vocab,
                              pt=wmt.bert_pretokenizer,
                              tk=wmt.bert_tokenizer,
                              tg=wmt.hantatagger)
    )

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

    model = AutoModelForMaskedLM.from_pretrained("bert-base-german-cased")

    # model: https://huggingface.co/transformers/v4.5.1/main_classes/model.html#transformers.PreTrainedModel.resize_token_embeddings

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    model = trainer.model
