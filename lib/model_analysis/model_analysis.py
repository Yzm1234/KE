## import packages
import os
import pandas as pd
from catboost import CatBoostClassifier, Pool
import numpy as np
import shap
import pickle
import csv
import json


def get_feature_importance(model_params, model_path, test_set_path,
                           shap_values_path=None,
                           output_file="feature_importance.tsv",
                           description_path="/global/cfs/cdirs/kbase/KE-Catboost/ziming/GO/Catboost/data/"
                                                    "go_terms/go_terms_description_category.json"):
    """
    This method takes model and test dataset, return testset feature importance in tsv table
    :param model_params: model params
    :type model_params: python dict object
    :param model_path: path to model file (json format)
    :type model_path: string
    :param test_set_path: path to test dataset (pickle format)
    :type test_set_path: string
    :param shap_values_path: path to shap values (pickle format) if none will calculate shap values based on model
                             and test set
    :type shap_values_path: string
    :param output_file: output file name, default is "feature_importance.tsv"
    :type output_file: string
    :param description_path: go term description path
    :type description_path: string
    :return: None
    :rtype: None
    """
    # load model
    model = CatBoostClassifier(model_params)
    model.load_model(model_path, format='json')

    # load test dataset
    test_set = pd.read_pickle(test_set_path)
    X_test = test_set[test_set.columns[1:]]
    y_test = test_set['biome']
    test_Pool = Pool(
        data=X_test,
        label=y_test,
        cat_features=np.where(X_test.dtypes != np.float)[0],
    )
    # load shap values
    if os.path.exists(shap_values_path):
        with open(shap_values_path, 'rb') as f:
            shap_values = pickle.load(f)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test, y_test)
        with open('shap_values.pkl', 'wb') as f:
            pickle.dump(shap_values, f)

    # calculate feature importance
    unsorted_feature_importance_scores = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
    feature_order_idx = np.argsort(unsorted_feature_importance_scores)[::-1]  # from the highest to the lowest
    sorted_feature_importance_scores = unsorted_feature_importance_scores[feature_order_idx]
    sorted_features = X_test.columns[feature_order_idx]
    feature_names = X_test.columns

    # read feature description
    with open(description_path, "r") as f:
        description = json.load(f)

    # write feature importance table
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(['GO', 'description', 'category', 'importance'])
        for i in range(len(list(sorted_features))):
            feature, importance_score = sorted_features[i], sorted_feature_importance_scores[i]
            feature_desc = description.get(feature, None)
            if feature_desc:
                desc, cate = feature_desc['description'].get('4.1', None), feature_desc['category']
                if desc:
                    desc = desc[0]
            else:
                desc, cate = None, None
            writer.writerow([feature, desc, cate, "{:.2f}".format(importance_score)])
