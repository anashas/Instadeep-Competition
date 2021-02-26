# The Approach used in this Competition is as following:
## [Competition website](https://zindi.africa/competitions/instadeep-enzyme-classification-challenge)

## [My Zindi Profile](https://zindi.africa/users/data_scientist)

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
   * small improvement from 88.99 to 89.15
* After making predictions, add the predicted test samples to the original training data, then re-train again.
* submit prediction.
* put your data into input folder, and your models into models folder.
* App demo using GradioML:

![alt text](https://github.com/anashas/Instadeep-Competition/blob/master/screenshot.png "Logo Title Text 1")
