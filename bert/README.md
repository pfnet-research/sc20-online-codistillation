online-codistillation-BERT
===

This repository provides artifacts for SC20 paper about online codistillation.
This implementation of BERT is based on the [NVIDIA implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT).


Dataset Preparation
---

[The quick start guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#quick-start-guide) in the NIVIDIA's BERT implementation shows how to prepare the pretraining and finetuning dataset.

In addition to the above instruction, this repository requires to create zip file containing the dataset.
For example:

```bash
$ zip -r books_wiki_en_corpus_phase1.zip ./hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training 
$ zip -r books_wiki_en_corpus_phase2.zip ./hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5_shard_1472_test_split_10/books_wiki_en_corpus/training
$ zip -r squad.zip ./squad
```


Training
---

`scripts/*.bash` will launch pretraining and finetuning.
For example, the following command launches the pretraining with the original online-codistillation:

```bash
$ mpiexec -n ${NUM_GPUS} -x MASTER_ADDR=<...> ./scripts/run_pretraining_with_original.bash
```
