import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras import utils



if __name__ == '__main__':
    df = pd.read_csv('inputs/Train.csv')
    df['kfold'] = -1
    
    df = df.sample(frac=1).reset_index(drop=True)

    lbl = LabelEncoder()
    df.loc[:,'LABEL'] = lbl.fit_transform(df.LABEL.values)
    
    '''
    classes = np.unique(df["LABEL"].values.tolist())
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    print(class_to_idx)
    idx_to_class = {val:key for key ,val in class_to_idx.items()}
    df['LABEL'].replace(class_to_idx, inplace=True)
    '''
    y = df.LABEL.values

    kf = StratifiedKFold(n_splits=5)

    for fold, (train_idx,val_idx) in enumerate(kf.split(X=df,y=y)):
        df.loc[val_idx, 'kfold'] = fold

    df.to_csv('inputs/stratified_5_folds.csv',index=False)    