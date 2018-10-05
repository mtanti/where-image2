# where-image2
Code used by the paper "[Where to put the Image in an Image Caption Generator](https://www.cambridge.org/core/journals/natural-language-engineering/article/where-to-put-the-image-in-an-image-caption-generator/A5B0ACFFE8E4AEAA5840DC61F93153F3)", which was accepted for Natural Language Engineering Special Issue on Language for Images. Code for previous version of paper (prior to review) is still available here https://github.com/mtanti/where-image.

This paper is a comparative evaluation of different ways to incorporate an image into an image caption generator neural network.

Works on both Python 2 and Python 3 (except for the MSCOCO evaluation toolkit which requires python 2).

## Dependencies

Python dependencies (install all with `pip`):

* `tensorflow` (v1.2)
* `future`
* `numpy`
* `scipy`
* `h5py`
* `hyperopt` (v0.1) (for hyperparameter tuning)
* `GPyOpt` (v1.2.1) (for hyperparameter tuning)

## Before running

1. Download [Karpathy's](http://cs.stanford.edu/people/karpathy/deepimagesent/) Flickr8k, Flickr30k, and MSCOCO datasets (including image features).
1. Download the [MSCOCO Evaluation toolkit](https://github.com/tylin/coco-caption).
1. Open `config.py`.
  1. Set `debug` to True or False (True is used to run a quick test).
  1. Set `raw_data_dir` to return the directory to the Karpathy datasets (`dataset_name` is 'flickr8k', 'flickr30k', or 'mscoco').
  1. Set `mscoco_dir` to the directory to the MSCOCO Evaluation toolkit.

## File descriptions

File name    |    Description
---|---
`results`    |    Folder storing results of each architecture's evaluation. You can find generated captions in `results/*/generated_captions.txt` and each caption corresponds to the image file name given by the corresponding line in `results/imgs_*.txt`. There is also a matrix in `results/*/retrieved_images.txt` of rows equal to the number of captions and columns equal to the number of images which gives the log probability of each possible caption/image pair. `multimodal_diffs_results_*.txt` files are described below.
`hyperparams`    |    Folder storing results of the hyperparameter tuning.
`hyperparam_phase1.py` (main)    |    Used to evaluate different hyperparameters on each architecture. Will save results in `hyperparams`. Delete `hyperparams/completed.txt` before running.
`hyperparam_phase2.py` (main)    |    Used to fine-tune the best hyperparameters found by `hyperparam_phase1.py`. Will save results in `hyperparams`. Delete `hyperparams/completed2.txt` before running.
`experiment.py` (main)    |    Used to run the actual experiments which evaluate each architecture. Will save results in `results`. Delete `results/results.txt` before running. If you have fine-tuned hyperparameters using `hyperparam_phase1.py` or `hyperparam_phase2.py` then you will first need to copy the hyperparameters found from `hyperparams/result_*.txt` to `config.py`.
`multimodal_vector_diffs.py` (main)    |    Used to measure the ability of each architecture to remember the image information as the caption gets generated. Will save results in `results/multimodal_diffs_results_*.txt` where '&ast;' is the caption length used.
`config.py` (library)    |    Configuration file containing hyperparameters, directories, and other settings.
`data.py` (library)    |    Functions and classes that deal with handling the datasets.
`helper_datasources.py` (library)    |    Functions and classes that simplify loading datasets.
`lib.py` (library)    |    General helper functions and classes.
`model_base.py` (library)    |    Super class for neural caption generation models that handles general applications such as beam search and sentence probability. This was created in order to facilitate creation of other model instatiations such as ensembles.
`model_idealmock.py` (library)    |    Neural caption generator that just memorises the test set and reproduces it (called `human` in the results). Used as a test for the generation and retrieval algorithms and as a ceiling for the caption diversity measures.
`model_normal.py` (library)    |    Neural caption generator with the actual architectures being tested.
`results.xlsx` (processed data)    |    MS Excel spreadsheet with the results of `experiments.py`.
`results_memory.xlsx` (processed data)    |    MS Excel spreadsheet with the results of `multimodal_vector_diffs.py`.

## Results

Descriptions of each column in `results/results.txt` and `results.xlsx`. 'Generated captions' refers to captions that were generated for test set images.

Column name    |    Description
---|---
`dataset_name`    |    The dataset used (Flickr8k, Flickr30k, or MSCOCO).
`architecture`    |    The architecture being tested (init, pre, par, merge, or human).
`run`    |    Each architecture is trained 3 separate times and the average of each result is taken. This column specifies the run number (1, 2, or 3).
`vocab_size`    |    The number of different word types in the vocabulary (all words in the training set that occur at least 5 times).
`num_training_caps`    |    The number of captions in the training set.
`mean_training_caps_len`    |    The mean caption length in the training set.
`num_params`    |    The number of parameters (weights and biases) in the architecture.
`geomean_pplx`    |    The geometric mean of the perplexity of all test set captions (given the image).
`num_inf_pplx`    |    The number of test set captions that resulted in infinity and were ignored from the geometric mean (occurs when at least one word has a probability of 0).
`vocab_used`    |    The number of words from the vocabulary that were used to generate all the captions.
`vocab_used_frac`    |    The fraction of vocabulary words that were used to generate all the captions.
`mean_cap_len`    |    The mean caption length of the generated captions.
`num_existing_caps`    |    The number of generated captions that were found somewhere in the training set as-is.
`num_existing_caps_frac`    |    The fraction of generated captions that were found somewhere in the training set as-is.
`existing_caps_CIDEr`    |    The CIDEr score of the generated captions that were found somewhere in the training set as-is (used to check if in the case where captions were being parroted, at least they were correct).
`unigram_entropy`    |    The entropy of unigram (word) frequencies in the generated captions.
`bigram_entropy`    |    The entropy of bigram (two adjacent words) frequencies in the generated captions.
`CIDEr`    |    The CIDEr score of the generated captions (generated by the MSCOCO Evaluation toolkit).
`METEOR`    |    The METEOR score of the generated captions (generated by the MSCOCO Evaluation toolkit).
`ROUGE_L`    |    The ROUGE L score of the generated captions (generated by the MSCOCO Evaluation toolkit).
`Bleu_1`    |    The BLEU-1 score of the generated captions (generated by the MSCOCO Evaluation toolkit).
`Bleu_2`    |    The BLEU-2 score of the generated captions (generated by the MSCOCO Evaluation toolkit).
`Bleu_3`    |    The BLEU-3 score of the generated captions (generated by the MSCOCO Evaluation toolkit).
`Bleu_4`    |    The BLEU-4 score of the generated captions (generated by the MSCOCO Evaluation toolkit).
`R@1`    |    The number of correct images that were the most relevant to their corresponding caption among all other images.
`R@5`    |    The number of correct images that were among the top 5 most relevant to their corresponding caption among all other images.
`R@10`    |    The number of correct images that were among the top 10 most relevant to their corresponding caption among all other images.
`median_rank`    |    The median rank of correct images when sorted by their relevance to their corresponding caption among all other images.
`R@1_frac`    |    The fraction of correct images that were the most relevant to their corresponding caption among all other images.
`R@5_frac`    |    The fraction of correct images that were among the top 5 most relevant to their corresponding caption among all other images.
`R@10_frac`    |    The fraction of correct images that were among the top 10 most relevant to their corresponding caption among all other images.
`median_rank_frac`    |    The median rank of correct images when sorted by their relevance to their corresponding caption among all other images,  divided by the number of images.
`num_epochs`    |    The number of epochs needed to train the model, before the perplexity on the validation set started to degrade.
`training_time`    |    The number of seconds needed to train the model.
`total_time`    |    The number of seconds needed to train and evaluate the model.
