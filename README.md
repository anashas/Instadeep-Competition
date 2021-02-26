# The Approach used in this Competition is as follows:
## [Competition website](https://zindi.africa/competitions/instadeep-enzyme-classification-challenge)

## [My Zindi Profile](https://zindi.africa/users/data_scientist)

1. Split the training file into 5 folds (source/train_folds.py)
2. Tokenize and padd the data.
3. Embedding layer( embedding dim = 128, max_len=300, batch_size=4096,learning_rate=0.001).
4. Two Bidirectional GRU layers with 300 units each.
5. One Conv1D layers with 300 filters.
6. One GlobalAveragePooling1D.
7. One GlobalMaxPooling1D.
8. Concatenate the max_pool and avg_pool
9. Dense layer with 1024 units
10. Batchnormalization
11 Dropout
12 Dense layer with 20 units
13. Check notebooks/enzyme.ipynb
14. Pseudo_labeling:
   ⋅⋅* Small improvement from 88.99 to 89.15
15. After making predictions, add the predicted test samples to the original training data, then re-train again.
16. Submit prediction.
17. Put your data into input folder, and your models into models folder.
18. The model was trained using Google Colab TPU
19. App demo using GradioML:

![alt text](https://github.com/anashas/Instadeep-Competition/blob/master/screenshot.png "Logo Title Text 1")
