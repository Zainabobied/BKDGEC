# BKDGEC: An automated framework to correct Arabic grammatical errors using deep learning. 
<!--This project introduces a Grammatical Error Correction (GEC) framework as a machine translation task for low-resource languages applied to the Arabic language. Initially, we propose a semi-supervised confusion method, named Equal Distribution of Synthetic Errors (EDSE), to generate wide parallel training data with a high diversity of training patterns. Besides, we propose Bidirectional Knowledge Distillation for Grammatical Error Correction (BKDGEC), which exploits two decoders: a forward decoder right-to-left and a backward decoder left-to-right into a regularization method. This takes advantage of leveraging the backward decoder’s information about the longer-term future and distilling knowledge learned in the backward decoder, which could encourage auto-regressive GEC models to plan ahead. Both decoders were trained into a joint training process and Kullback-Leibler divergence was applied to measure the agreement between both decoders as a regulation term.-->
# Model requirements
Regarding load and run the trained models it requires a working installation of following:
- Python 3.6.10 or latest 
- pytorch==1.6.0
- torchtext==0.6.0
- numpy==1.17.0
- pandas==1.17.0
- PyArabic==0.6.10
- bpemb==0.3.2
- nltk==3.3
