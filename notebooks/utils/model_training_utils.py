import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import multiprocess
from itertools import product

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error


def load_features(folder_name, feature_subset="combined"):
    """
    Load features from a folder into a dictionary of dataframes.
    
    Args:
        folder_name (str): path to the folder containing the features
        feature_subset (str): subset of features to load

    Returns:
        feat_dict (dict): dictionary of dataframes containing the features
    """    
    # Create a list of files in the folder
    files = os.listdir(folder_name)

    feat_dict = {}
    # Loop through the files and read them into the dataframe
    for file in tqdm(files):
        if not file.endswith(".parquet"):
            continue
        window_size = int(file.split("_")[0])
        feat_df = pd.read_parquet(folder_name + file)

        # Filter out the features that are not in the feature subset
        if feature_subset != "combined":
            feat_df = feat_df[['misalignment', 'recording_nr', 'direction', 'speed', 'window_id'] + [col for col in feat_df.columns if col.split("__")[0] == feature_subset]]
        feat_dict[window_size] = feat_df
    return feat_dict


def stratify_df(df, stratify_labels: list, n_splits: int = 5):
    """
    Stratify a dataframe based on multiple labels.

    Args:
        df (pd.DataFrame): dataframe to be stratified
        stratify_labels (list): list of labels to stratify on
        n_splits (int): number of splits

    Returns:
        split_dict (dict): dictionary of dataframes containing the stratified data
    """

    split_dict = {fold_nr: {'train':[], 'test':[]} for fold_nr in range(1, n_splits + 1)}
    # Groupby misalignment, direction, speed, and window_id
    for name, group in df.groupby(stratify_labels):
        # Groupby recording_nr
        for rec_name, rec_group in group.groupby('recording_nr'):
            for fold_nr in split_dict.keys():
                # Add train and test set to each fold
                if fold_nr == int(rec_name) or fold_nr == int(rec_name) - 5:
                    split_dict[fold_nr]['test'].append(rec_group)
                else:
                    split_dict[fold_nr]['train'].append(rec_group)
    
    # Concat the lists of dataframes
    for fold_nr, set_dict in split_dict.items():
        split_dict[fold_nr]['train'] = pd.concat(set_dict['train'])
        split_dict[fold_nr]['test'] = pd.concat(set_dict['test'])

    return split_dict


def cross_val_catboost(split_dict, cur_feature_set):
    maes = []
    for fold_nr, set_dict in split_dict.items():
        x_train = set_dict['train'].drop(['misalignment', 'recording_nr', 'direction', 'speed', 'window_id'], axis=1)[cur_feature_set]
        x_test = set_dict['test'].drop(['misalignment', 'recording_nr', 'direction', 'speed', 'window_id'], axis=1)[cur_feature_set]
        y_train = set_dict['train']['misalignment']
        y_test = set_dict['test']['misalignment']
        model = CatBoostRegressor(iterations=1000, verbose=0, od_wait=50, od_type="Iter", random_state=4, thread_count=6)
        model = model.fit(x_train, y_train, eval_set=(x_test, y_test))
        preds = model.predict(x_test)
        maes += [mean_absolute_error(preds, y_test)]
    return np.mean(maes)


def select_features(feature_names, split_dict, score_improvement_threshold=0.00025):
    selected_features = []
    scores = []
    improvement = True
    score = float('inf')

    features = set(feature_names)

    prev_score = score
    best_score = score
    while improvement:
        best_feature, improvement = None, False
        best_score = float('inf')

        # Create a closure
        def calc_score(feature):
            cur_feature_set = sorted(list(set(selected_features).union({feature})))
            try:
                return cross_val_catboost(split_dict, cur_feature_set)
            except:
                print(f"Error with feature {feature}")
                return float('inf')

        features_ = sorted(list(features))

        with multiprocess.Pool(processes=6) as pool:
            results = list(tqdm(pool.imap(calc_score, features_), total=len(features_)))

        for score, feature in zip(results, features_):
            if score < best_score and score < prev_score - score_improvement_threshold:
                best_score = score
                best_feature = feature
                improvement = True
                
        if best_feature is not None:
            prev_score = best_score
            selected_features.append(best_feature)
            scores.append(best_score)
            features = features.difference({best_feature})
            
        print(improvement, best_score, best_feature)

    return selected_features, scores


def train_model(window_size, split_dict, sel_feat_names):
    """
    Train a catboost model.

    Args:
        window_size (int): window size
        split_dict (dict): dictionary of dataframes containing the stratified data
        sel_feat_names (list): list of selected features
        
    Returns:
        pipe (CatBoostRegressor): trained catboost model
    """
    print("Window size: ", window_size)
    train_mae_list = []
    test_mae_list = []
    pipe_list = []
    for fold_nr, set_dict in tqdm(split_dict.items()):
        x_train = set_dict['train'].drop(['misalignment', 'recording_nr', 'direction', 'speed', 'window_id'], axis=1)[sel_feat_names]
        x_test = set_dict['test'].drop(['misalignment', 'recording_nr', 'direction', 'speed', 'window_id'], axis=1)[sel_feat_names]
        y_train = set_dict['train']['misalignment']
        y_test = set_dict['test']['misalignment']

        pipe = CatBoostRegressor(iterations=1000, verbose=0, od_wait=50, od_type="Iter", random_state=4)
        pipe = pipe.fit(x_train, y_train, eval_set=(x_test, y_test))
        pipe_list.append(pipe)

        train_preds = pipe.predict(x_train)
        train_mae = mean_absolute_error(y_train, train_preds)
        train_mae_list.append(train_mae)

        test_preds = pipe.predict(x_test)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_mae_list.append(test_mae)

    print("MAE over train folds:", np.mean(train_mae_list), "+/-", np.std(train_mae_list)) 
    print("MAE over test folds:", np.mean(test_mae_list), "+/-", np.std(test_mae_list))
    print()
    return pipe_list


def stratify_perm_df(df, stratify_labels: list, output_labels: list, n_splits: int = 5):
    """
    Stratify a dataframe based on multiple labels.

    Args:
        df (pd.DataFrame): dataframe to be stratified
        stratify_labels (list): list of labels to stratify on.
        output_labels (list): list of labels to be outputted
        n_splits (int): number of splits

    Returns:
        split_dict (dict): dictionary of dataframes containing the stratified data
    """
    # Check if output labels are in stratify labels
    for label in output_labels:
        if label not in stratify_labels:
            raise ValueError(f"Label {label} is not in stratify labels.")


    config_list = product(*[df[config].unique() for config in output_labels])
    split_dict = {config:{fold_nr: {'train':[], 'test':[]} for fold_nr in range(1, n_splits + 1)} for config in config_list}


    # Groupby misalignment, direction, speed, set and window_id
    for name, group in df.groupby(stratify_labels):
        # Groupby recording_nr
        for rec_name, rec_group in group.groupby('recording_nr'):
            label_tuple = tuple(rec_group[label].values[0] for label in output_labels)
            for fold_nr in split_dict[label_tuple].keys():
                # Add train and test set to each fold
                if fold_nr == int(rec_name) or fold_nr == int(rec_name) - 5:
                    split_dict[label_tuple][fold_nr]['test'].append(rec_group)
                else:
                    split_dict[label_tuple][fold_nr]['train'].append(rec_group)

    # Concat the lists of dataframes
    for config, fold_dict in split_dict.items():
        for fold_nr, set_dict in fold_dict.items():
            split_dict[config][fold_nr]['train'] = pd.concat(set_dict['train'])
            split_dict[config][fold_nr]['test'] = pd.concat(set_dict['test'])
    return split_dict


def train_perm_model(window_size, split_dict, sel_feat_names):
    """
    Train a catboost model.

    Args:
        window_size (int): window size
        split_dict (dict): dictionary of dataframes containing the stratified data
        sel_feat_names (list): list of selected features
        
    Returns:
        pipe (CatBoostRegressor): trained catboost model
    """
    print("Window size: ", window_size)
    pipe_dict = {}
    for config, fold_dict in tqdm(split_dict.items()):
        print("Config: ", config)
        pipe_list = []
        train_mae_list = []
        test_mae_list = []
        for fold_nr, set_dict in tqdm(fold_dict.items()):
            x_train = set_dict['train'].drop(['misalignment', 'recording_nr', 'direction', 'speed', 'window_id'], axis=1)[sel_feat_names[config]['selected_features']]
            x_test = set_dict['test'].drop(['misalignment', 'recording_nr', 'direction', 'speed', 'window_id'], axis=1)[sel_feat_names[config]['selected_features']]
            y_train = set_dict['train']['misalignment']
            y_test = set_dict['test']['misalignment']

            pipe = CatBoostRegressor(iterations=1000, verbose=0, od_wait=50, od_type="Iter", random_state=4)
            pipe = pipe.fit(x_train, y_train, eval_set=(x_test, y_test))
            pipe_list.append(pipe)

            train_preds = pipe.predict(x_train)
            train_mae = mean_absolute_error(y_train, train_preds)
            train_mae_list.append(train_mae)

            test_preds = pipe.predict(x_test)
            test_mae = mean_absolute_error(y_test, test_preds)
            test_mae_list.append(test_mae)
        pipe_dict[config] = pipe_list
            
        print("MAE over train folds:", np.mean(train_mae_list), "+/-", np.std(train_mae_list)) 
        print("MAE over test folds:", np.mean(test_mae_list), "+/-", np.std(test_mae_list))
        print()
    return pipe_dict