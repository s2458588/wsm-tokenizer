import datasets
import wm_tokenizer as wm
from HanTa import HanoverTagger as ht
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
tagger = ht.HanoverTagger('morphmodel_ger.pgz')


def wordmap2tokenizer(
        sent: str,
        pos_tag: str,
        vocab,
        tk=tokenizer,
        tg=tagger,
):
    """Takes list of strings, a STTS Pos-tag and a vocabulary: dict/list."""
    encoding = []

    for i in sent.split(" "):
        if tg.analyze(i)[1].startswith(pos_tag):
            wmt = wm.SequenceTokenizer(vocab=vocab, target=i)
            tokenizer_format = [wmt.maxed[0]]+["##" + i for i in wmt.maxed[1:]]
            tokenizer.add_tokens(tokenizer_format)
            encoding.extend(tokenizer_format)
        else:
            encoding.extend(tk.tokenize(i))
    return tokenizer.encode_plus(encoding[1:-1], return_token_type_ids=True, return_attention_mask=True, padding=True, )


def main():
    dataset = datasets.Dataset = ...
    dataset = dataset.map(
        tokenize()
    )

    trainer = Trainer(

    )

    model = trainer.model
