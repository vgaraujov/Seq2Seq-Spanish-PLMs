# Seq2Seq Spanish Pre-trained Language Models
This repository contains the models and scripts from the paper [Sequence-to-Sequence Spanish Pre-trained Language Models](https://arxiv.org/abs/2309.11259).

### Models
All our pre-trained models can be found on the [HuggingFace Hub](https://huggingface.co/collections/vgaraujov/sequence-to-sequence-spanish-plms-65f87d42c1823e23cf863b34).

[BARTO](https://huggingface.co/vgaraujov/bart-base-spanish) and [T5S](https://huggingface.co/vgaraujov/t5-base-spanish) are variants of [BART](https://arxiv.org/abs/1910.13461) and [T5](https://arxiv.org/abs/1910.10683), respectively, pre-trained exclusively on Spanish corpora in a self-supervised manner. BARTO and T5S are base-sized versions comprising approximately 140 million and 220 million parameters, respectively.

You can load T5S like this:
```
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("vgaraujov/t5-base-spanish")
model = AutoModel.from_pretrained("vgaraujov/t5-base-spanish")
```
You can load BARTO like this:
```
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("vgaraujov/bart-base-spanish")
model = AutoModel.from_pretrained("vgaraujov/bart-base-spanish")
```

### Additional Models
[LEDO](https://huggingface.co/vgaraujov/led-base-16384-spanish) was built to process sequences longer sequences by leveraging the weights of BARTO. To process 16K tokens, BARTO's position embedding matrix was copied 16 times.

You can load LEDO like this:
```
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("vgaraujov/led-base-16384-spanish")
model = AutoModel.from_pretrained("vgaraujov/led-base-16384-spanish")
```

BERT2BERT-style models were introduced as baselines. By leveraging [Encoder Decoder Models](https://huggingface.co/docs/transformers/en/model_doc/encoder-decoder) from Huggingface and using [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) and [RoBERTa-BNE](https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne) checkpoints, we initialized BETO2BETO and RoBERTa2RobERTa.

You can load BETO2BETO like this:
```
from transformers import EncoderDecoderModel

model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "dccuchile/bert-base-spanish-wwm-cased",
    "dccuchile/bert-base-spanish-wwm-cased",
    tie_encoder_decoder=False
)
```
Note: `tie_encoder_decoder=True` initializes BETOShare or RoBERTaShare.

### Fine-tuning
To fine-tune BARTO, T5S, and LEDO, we rely on HuggingFace examples for [summarization](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) and [translation](https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation).

For tasks like generative question-answering, split-and-rephrase, and dialogue, we implemented additional scripts found in this repository.

Additionally, we implemented the script versions to experiment with BERT2BERT-style models, which are also found in this repository.

We include experiment files that you can run to replicate our results. For example, running:
```
bash run_summarization.sh
```

### Citation

If you find this repository useful for your research, please consider citing our paper: 
```
@misc{araujo2024sequencetosequence,
      title={Sequence-to-Sequence Spanish Pre-trained Language Models}, 
      author={Vladimir Araujo and Maria Mihaela Trusca and Rodrigo Tufiño and Marie-Francine Moens},
      year={2024},
      eprint={2309.11259},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
