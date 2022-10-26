import argparse
import joblib
import json
import os
import pickle
from pathlib import Path

import catboost as cb
import optuna
import pandas as pd
from catboost import Pool
from optuna.visualization import plot_contour
from optuna.visualization import plot_parallel_coordinate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# add arg parser
parser = argparse.ArgumentParser(description="Catboost model hyper-parameter selection")
parser.add_argument("study_name", help="name of study")
parser.add_argument("input_file", help="input file path")
parser.add_argument("output_folder", help="output folder path")
parser.add_argument("sampler", help="Optuna sampler", choices=["RandomSampler", "TPESampler", "CmaEsSampler"])
parser.add_argument("-k", "--kfold", default=5, type=int, help="k fold validation")
parser.add_argument("-t", "--trial_number", default=100, type=int, help="total number of trials in an Optuna study")
parser.add_argument("-r", "--resume", action='store_true',
                    help="if resume from an existing study. It will complete the trial number if not finished by previous study")
global args
args = parser.parse_args()


def objective(trial: optuna.Trial):
    file_path = args.input_file
    df = pd.read_pickle(file_path)
    X = df.iloc[:, 1:]
    y = df['biome']
    skf = StratifiedKFold(n_splits=args.kfold)
    acc_test = []
    for train_val, test in skf.split(X, y):
        X_train_val, y_train_val = X.iloc[train_val], y.iloc[train_val]
        X_test, y_test = X.iloc[test], y.iloc[test]
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, random_state=42, test_size=0.22,
                                                          stratify=y_train_val)

        # 1. train pool
        X_train_pool = Pool(
            data=X_train,
            label=y_train,
        )
        X_val_pool = Pool(
            data=X_val,
            label=y_val,
        )
        # 2. Init the model
        params = {
            'depth': trial.suggest_int("depth", 4, 10, step=2),  # Maximum tree depth is 16
            'learning_rate': trial.suggest_float("learning_rate", 0.1, 0.3, step=0.05),
            # 'l2_leaf_reg': trial.suggest_int("l2_leaf_reg", 2, 4, step=1),
            # 'random_strength': trial.suggest_int("random_strength", 1, 5, step=1),
            # 'bagging_temperature': trial.suggest_int("bagging_temperature", 0, 5, step=1),
        }
        gbm = cb.CatBoostClassifier(
            custom_metric='Accuracy',
            random_seed=42,
            task_type="GPU",
            **params)

        # 3. training
        gbm.fit(X_train_pool, eval_set=X_val_pool, verbose=0, early_stopping_rounds=100)

        # 4. predict
        preds = gbm.predict(X_test)
        acc = accuracy_score(y_test, preds)
        acc_test.append(acc)

    return sum(acc_test) / args.kfold


def ke_optuna(study_name, sampler, n_trials, output_dir):
    # select sampler
    if sampler == "RandomSampler":
        optuna_sampler = optuna.samplers.RandomSampler()
    elif sampler == "TPESampler":
        optuna_sampler = optuna.samplers.TPESampler()
    elif sampler == "CmaEsSampler":
        optuna_sampler = optuna.samplers.CmaEsSampler()

    os.chdir(output_dir)
    storage_name = "sqlite:///{}.db".format(study_name)
    if args.resume:
        # resume from existing study
        print("Loading previous study...")
        study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                                    sampler=optuna_sampler)
        print("Loading finished.")
        if len(study.trials) < args.trial_number:
            n_trials = args.trial_number - len(study.trials)
            print("Loaded study has completed {} trials, will continue another {} trials ".format(len(study.trials),
                                                                                                  n_trials))

        else:
            print("Loaded study has completed all {} trials, the optimization is done.".format(args.trial_number))
            return
    else:
        if os.path.exists("{}.db".format(study_name)):
            os.remove("{}.db".format(study_name))
        # create a new study
        study = optuna.create_study(study_name=study_name, storage=storage_name, sampler=optuna_sampler)

    study.optimize(objective, n_trials=n_trials)

    # save best params to json file
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as fp:
        json.dump(study.best_params, fp)

    # save parallel plot of result
    parallel_plot = plot_parallel_coordinate(study)
    parallel_plot.write_image(os.path.join(output_dir, "{}_parallel_plot.png".format(study_name)))
    parallel_plot.show()

    # save contour plot of result
    contour_plot = plot_contour(study)
    contour_plot.write_image(os.path.join(output_dir, "{}_contour_plot.png".format(study_name)))
    contour_plot.show()


if __name__ == "__main__":
    output_dir = os.path.join(args.output_folder, "{}_output/".format(args.study_name))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ke_optuna(args.study_name, args.sampler, args.trial_number, output_dir)