# knowledge-graph-recommender
Comps replication repository on knowledge graphs for recommendation

## General Usage Information

### Jupyter notebooks to load song data and construct KG:
Run cells in song_data_prep.ipynb located in data/song_data

Run cells in find_dense_sparse_subnetworks.ipynb located in data/song_data

Run cells in train_test_split.ipynb located in data/song_data

Run cells in id_mapping_prep.ipynb located in data/song_data_vocab

### recommender.py command line arguments
`--train` to train model, `--eval` to evaluate

`--find_paths` if you want to find paths before training or evaluating

`--train_path_file` and `--test_path_file` designate the file to save/load paths from

`--train_inter_limit` and `--test_inter_limit` designate the max number of positive train/test interactions to find paths for

`--model` designates the model to train or evaluate from

`--load_checkpoint` if you want to load a model checkpoint (weights and parameters) before training

`-b` designates model batch size and `-e` number of epochs to train model for

### Training syntax
Command line syntax to find train paths and train model on first 10000 positive interactions:

Note: each positive interaction is paired with 4 negative ones.

`python3 recommender.py --train --find_paths --train_inter_limit 10000 --train_path_file train_interactions_10000.txt --model model_10000.pt -e 10`

### Evaluating syntax
Command line syntax to find test paths and evaluate trained model(the 10000 interaction saved model) on 1000 interaction groups:

Note: Each interaction group is 1 positive interaction paired with 100 negative interactions.

`python3 recommender.py --eval --find_paths --test_inter_limit 1000 --test_path_file test_interactions_1000.txt --model model_10000.pt`
