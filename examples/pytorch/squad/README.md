
## SQuAD

SQuAD 1.1 dataset:

* [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
* [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
* [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

Run save of dataset with pre-processing
```bash
python run_preprocess_squad.py \
    --tokenizer_name bert-base-uncased \
    --do_lower_case \
    --train_file train-v1.1.json \
    --predict_file dev-v1.1.json \
    --do_train --do_eval --threads 16
```

After finish save of dataset, start training SQuAD with saving model and optimizer
```bash
python run_squad.py \
    --model_type bert \
    --output_dir /tmp/output \
    --model_name_or_path bert-base-uncased \
    --do_lower_case \
    --do_train --evaluate_during_training
```


SQuAD 2.0 dataset:

- [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
- [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
- [evaluate-v2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

Run save of dataset with pre-processing
```bash
python run_preprocess_squad.py \
    --tokenizer_name bert-base-uncased \
    --do_lower_case \
    --version_2_with_negative \
    --train_file train-v1.1.json \
    --predict_file dev-v1.1.json \
    --do_train --do_eval --threads 16
```

After finish save of dataset, start training SQuAD with saving model and optimizer
```bash
python run_squad.py \
    --model_type bert \
    --output_dir /tmp/output \
    --version_2_with_negative \
    --model_name_or_path bert-base-uncased \
    --do_lower_case \
    --do_train --evaluate_during_training
```