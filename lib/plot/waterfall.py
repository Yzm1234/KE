import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap


class WaterFall:
    def __init__(self, model, test_set):
        self.model = model
        self.test_set = test_set
        self.test_set.reset_index(drop=True, inplace=True)

    @staticmethod
    def read_feature_description(go_term_description_path):
        # go_term_description_path = "/global/cfs/cdirs/kbase/KE-Catboost/ziming/GO/data/go_terms/go_terms_description_category.json"
        with open(go_term_description_path, "r") as f:
            go_term_description = json.load(f)
        return go_term_description

    @staticmethod
    def get_feature_description(go_term_description, go_term_list):
        feature_list_with_desc = []
        for feature in go_term_list:
            feature_desc = go_term_description.get(feature, None)
            if feature_desc:
                desc, cate = feature_desc['description'].get('4.1', None), feature_desc['category']
                if desc:
                    desc = desc[0]
            else:
                desc, cate = None, None
            feature_list_with_desc.append([feature, desc, cate])
        return feature_list_with_desc

    def get_waterfall(self, groupby="prediction", output_path="output", top_n=10,
                      go_term_description_path="/global/cfs/cdirs/kbase/KE-Catboost/ziming/GO/Catboost/data/go_terms/go_terms_description_category.json"):
        # get X_test and u_test

        X_test = self.test_set[self.test_set.columns[1:]]
        y_test = self.test_set['biome']

        # get predictions
        y_pred = self.model.predict(X_test)

        # get explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test, y_test)

        # get shap values
        # shap_values = get_shap_values(model, X_test, y_test)

        if groupby == "prediction":  # group by samples prediction labels
            X_test.insert(0, "prediction", y_pred)
            avg_X = X_test.groupby(['prediction']).mean()
            labels = y_pred  # labels used to group samples

        elif groupby == "groud_truth":  # group b samples ground truth labels
            avg_X = self.test_set.groupby(['biome']).mean()
            labels = y_test

        Path(output_path).mkdir(exist_ok=True)
        for biome_index, biome in enumerate(self.model.classes_):
            biome_dir = os.path.join(output_path, biome)
            Path(biome_dir).mkdir(exist_ok=True)
            # plot
            samples_idxes = np.where(labels == biome)[0]  # return an array
            samples_avg_shap_values = np.mean(shap_values[biome_index][samples_idxes], axis=0)
            shap.waterfall_plot(shap.Explanation(samples_avg_shap_values,
                                                 base_values=explainer.expected_value[biome_index],
                                                 data=avg_X.iloc[biome_index],
                                                 feature_names=avg_X.columns.tolist()),
                                max_display=10,
                                show=False)
            fig = plt.gcf()
            fig.savefig(os.path.join(biome_dir, '{}_waterfall.png'.format(biome)), bbox_inches='tight')

            # tsv file
            soorted_importantce = sorted(samples_avg_shap_values, key=abs, reverse=True)[:top_n]
            sorted_features = avg_X.columns[np.argsort(abs(samples_avg_shap_values))[::-1]][:top_n]
            go_term_description = self.read_feature_description(go_term_description_path)
            sorted_features_with_description = self.get_feature_description(go_term_description, sorted_features)
            with open(os.path.join(biome_dir, 'top_features.tsv'), 'w') as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(['GO', "Description", "Category", "Importance"])
                for i in range(top_n):
                    go = sorted_features_with_description[i][0]
                    description = sorted_features_with_description[i][1]
                    category = sorted_features_with_description[i][2]
                    importance = soorted_importantce[i]
                    writer.writerow([go, description, category, importance])