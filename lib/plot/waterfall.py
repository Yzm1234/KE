import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


from catboost import CatBoostClassifier


# add arg parser
parser = argparse.ArgumentParser(description="Catboost model feature importance analysis")
parser.add_argument("model", help="CatBoost model path, json file")
parser.add_argument("test_dataset", help="test_dataset_path")
parser.add_argument("output_folder", help="output folder path")
parser.add_argument("-g", "--group_by", choices=["prediction", "ground_truth"],
                    help="the label by which to group samples for each biome")
parser.add_argument("-n", "--top_n", default=10, type=int, help="top n features plotted in waterfall plot")
parser.add_argument("-d", "--feature_description", action='store_true',
                    default="/global/cfs/cdirs/kbase/KE-Catboost/ziming/GO/data/go_terms/go_terms_description_category.json",
                    help="feature description path, now only available for GO data")

global args
args = parser.parse_args()


class WaterFall:
    def __init__(self, model, test_set):
        model_params = CatBoostClassifier(
            loss_function='MultiClass',
            custom_metric='Accuracy',
            learning_rate=0.15,
            random_seed=42,
            l2_leaf_reg=3,
            iterations=2000,
        )
        self.model = model_params.load_model(model, format='json')
        self.test_set = pd.read_pickle(test_set)
        self.test_set.reset_index(drop=True, inplace=True)

    @staticmethod
    def read_feature_description(description_path):
        with open(description_path, "r") as f:
            description = json.load(f)
        return description

    @staticmethod
    def get_feature_description(description, feature_list):
        feature_list_with_desc = []
        for feature in feature_list:
            feature_desc = description.get(feature, None)
            if feature_desc:
                desc, cate = feature_desc['description'].get('4.1', None), feature_desc['category']
                if desc:
                    desc = desc[0]
            else:
                desc, cate = None, None
            feature_list_with_desc.append([feature, desc, cate])
        return feature_list_with_desc

    def get_feature_importance(self, groupby="prediction", output_path="output", top_n=10,
                               add_description=False,
                               description_path="/global/cfs/cdirs/kbase/KE-Catboost/ziming/GO/Catboost/data/go_terms/go_terms_description_category.json"):

        # get X_test and y_test
        X_test = self.test_set[self.test_set.columns[1:]]
        y_test = self.test_set['biome']

        # get predictions
        y_pred = self.model.predict(X_test)

        # get explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test, y_test)

        if groupby == "prediction":  # group by samples prediction labels
            X_test.insert(0, "prediction", y_pred)
            avg_X = X_test.groupby(['prediction']).mean()
            labels = y_pred  # labels used to group samples

        elif groupby == "ground_truth":  # group b samples ground truth labels
            avg_X = self.test_set.groupby(['biome']).mean()
            labels = y_test

        print("reading feature descprition")
        # load feature description
        if add_description:
            description = self.read_feature_description(description_path)
            features_with_description_list = self.get_feature_description(description, avg_X.columns.tolist())
            features = [" | ".join(f) for f in features_with_description_list]
        else:
            features = avg_X.columns.tolist()

        Path(output_path).mkdir(exist_ok=True)
        for biome_index, biome in enumerate(self.model.classes_):
            print(biome)
            # create subdir for each class
            biome_dir = os.path.join(output_path, biome)
            Path(biome_dir).mkdir(exist_ok=True)

            print("\tDescription loaded, start ploting")
            # plot waterfall for each class
            samples_idxes = np.where(labels == biome)[0]  # return an array
            samples_avg_shap_values = np.mean(shap_values[biome_index][samples_idxes], axis=0)
            shap.waterfall_plot(shap.Explanation(samples_avg_shap_values,
                                                 base_values=explainer.expected_value[biome_index],
                                                 data=avg_X.iloc[biome_index],
                                                 feature_names=features),
                                max_display=top_n,
                                show=False)
            print("\tplotting")
            fig = plt.gcf()
            fig.savefig(os.path.join(biome_dir, '{}_waterfall.png'.format(biome)), bbox_inches='tight')
            plt.clf()
            print("\tcreating tsv file")
            # save feature importance table in tsv file
            sorted_importantce = sorted(samples_avg_shap_values, key=abs, reverse=True)
            sorted_features = avg_X.columns[np.argsort(abs(samples_avg_shap_values))[::-1]]

            with open(os.path.join(biome_dir, 'top_features.tsv'), 'w') as f:
                writer = csv.writer(f, delimiter="\t")
                if add_description:
                    sorted_features_with_description = [features_with_description_list[i] for i in
                                                        list(np.argsort(abs(samples_avg_shap_values))[::-1])]
                    writer.writerow(['Feature', "Description", "Category", "Importance"])
                    for i in range(len(sorted_features)):
                        feature = sorted_features_with_description[i][0]
                        description = sorted_features_with_description[i][1]
                        category = sorted_features_with_description[i][2]
                        importance = sorted_importantce[i]
                        writer.writerow([feature, description, category, importance])
                else:
                    writer.writerow(['Feature', "Importance"])
                    for i in range(len(sorted_features)):
                        feature = sorted_features[i]
                        importance = sorted_importantce[i]
                        writer.writerow([feature, importance])
            print("\ttsv file is done.")
        return


if __name__ == "__main__":
    wf = WaterFall(args.model, args.test_dataset)
    if args.feature_description:
        wf.get_feature_importance(groupby=args.group_by, output_path=args.output_folder, top_n=args.top_n,
                                  add_description=True,
                                  description_path=args.feature_description)
    else:
        wf.get_feature_importance(groupby=args.group_by, output_path=args.output_folder, top_n=args.top_n)

