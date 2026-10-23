# [Paper Review] Neural Machine Translation by Jointly Learning to Align and Translate 
**Authors:** Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio\
**Institutions:** Jacobs University Bremen, Université de Montréal\
**Publication:** ICLR 2025\
**Link:** https://arxiv.org/abs/1409.0473

## Abstract & Introduction  
Previous machine translation systems consisted of many smaller subcomponents. A new approach called _neural machine translation_ uses a single neural network to read a sentence and output a translation. Most neural machine translation models are encdoder-decoders where a sentence is transformed in a _fixed-length_ encoding vector, from which the translation is outputted. The problem with this approach is that it is difficult to compress all the information into a fixed-length vector for very long sentences. 

To address this issue, the paper proposes an "attention" mechanism that for each target word, identifies the most relevant words in the original sentence. Each new target word is produced based on its "context vectors" as well as the previously generated target words. 

## Methods 

## Results 

