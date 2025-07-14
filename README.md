## dial2vec Implementation


Heavy modification of [AlibabaResearch/DAMO-ConvAI dial2vec](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/dial2vec) for research use-case

- Uses MAC ARM 
- Only has code needed for my implementation 
- Uses ModernBERT over RoBERTa and BERT 

doc2dial dataset is required in `datasets/doc2dial` and can be found [here](https://drive.google.com/file/d/1KpxQGXg9gvH-2u21bAMykL5N-tpYU2Dr/view?usp=sharing)

## To-do/Plans

- Eventually utilize cuda once I have compute 
- Better processing of data / multiprocessing 
- Train on doc2dial and upload weights on huggingface (AliBabaResearch does not do this)
- doc2dialsequenceclassification (Stack classification head + finetune)
- Handle >2 speakers
- Notebook to show example of converting into .tsv format with random sorting for negative samples 

## Other 

Original paper: [Dial2vec: Self-Guided Contrastive Learning
of Unsupervised Dialogue Embeddings](https://aclanthology.org/2022.emnlp-main.490.pdf)


