The Approach used in this Competition is as following:

* Split the training file into 5 folds (source/train_folds.py)
* Tokenize and padd the data.
* Embedding layer( embedding dim = 128, max_len=300, batch_size=4096,learning_rate=0.001).
* Two Bidirectional GRU layers with 300 units each.
* One Conv1D layers with 300 filters.
* one GlobalAveragePooling1D.
* one GlobalMaxPooling1D.
* concatenate the max_pool and avg_pool
* dense layer with 1024 units
* batchnormalization
* dropout
* dense layer with 20 units
* check notebooks/enzyme.ipynb
* Pseudo_labeling:
* After making predictions, add the predicted test samples to the original training data, then re-train again
* submit prediction.
* put your data into input folder, and your models into models folder