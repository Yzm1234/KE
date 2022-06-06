import json
import numpy as np
import os
import plotly.figure_factory as ff
import plotly.io as pio

from collections import Counter
from collections import defaultdict
from pathlib import Path
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import fdrcorrection


class ClusteringTree:
    def __init__(self, data_df, _abs=True, output_path='output'):
        """
        Instance initialization
        :param data_df: dataset, first column is lable, e.g. "biome". Rest columns are features
        :type data_df: pandas data framework
        :param _abs: if using abs value for distance. If true, dist = 1 - abs(correlation)
        :type _abs: boolean
        :param output_path: output folder name
        :type output_path: string
        """
        self.data_df = data_df
        self.groupped_data_df = data_df.groupby(['biome']).mean()
        self.correlation_mat = self.groupped_data_df.T.corr()
        self.abs = _abs
        self.output_path = output_path
        Path(output_path).mkdir(exist_ok=True)

    def getZ(self):
        """
        Get Z. Z is a 2d array of shape N x 4. N is the number of clusters.
        4 is [cluster1_idx, cluster2_idx, distance, number of nodes in this cluster]
        :return: Z
        :rtype: 2d array
        """
        if self.abs:
            dissimilarity = 1 - abs(self.correlation_mat)
        else:
            # TODO
            # check later if this is 2
            dissimilarity = 1 - self.correlation_mat
        Z = linkage(squareform(dissimilarity), 'complete')
        return Z

    def biome_root_remover(self):
        """
        remove first "root" term from every biome
        :return: all unique biomes
        :rtype: list
        """
        biome_list = list(self.correlation_mat.columns)
        biome_list_rm_root = []
        total_terms = 0
        for biome in biome_list:
            b = biome.split(":")
            if len(b) == 1:
                total_terms += 1
                biome_list_rm_root.append(b[0])
            else:
                total_terms += len(b[1:])
                new_biome = ":".join([i.capitalize() for i in b[1:]])
                biome_list_rm_root.append(new_biome)
        return biome_list_rm_root

    def get_cluster_dict(self, Z, biome_list):
        """
        Z: [cluster1_idx, cluster2_idx, coordinate, coordinate]
        biome_list: all unique biomes after remove prefix "root"
        """
        nodes_dict = {}
        res = {}
        unique_biome_count = len(Z) + 1
        for iteration in range(len(Z)):
            new_cluster_id = iteration + len(Z) + 1
            cluster_1_id, cluster_2_id = int(Z[iteration][0]), int(Z[iteration][1])

            if cluster_1_id > unique_biome_count-1: # is a merged cluster
                cluster_1_nodes = nodes_dict[cluster_1_id] # is a list
            else:
                cluster_1_nodes = [cluster_1_id]

            if cluster_2_id > unique_biome_count-1:
                cluster_2_nodes = nodes_dict[cluster_2_id] # is a list
            else:
                cluster_2_nodes = [cluster_2_id]

            new_cluster_nodes = cluster_1_nodes + cluster_2_nodes
            nodes_dict[new_cluster_id] = new_cluster_nodes
        for k, v in nodes_dict.items():
            res[k-len(Z)] = [biome_list[i] for i in v]

        file_path = "cluster_biome_dict.json"
        with open(os.path.join(self.output_path, file_path), "w") as outfile:
            json.dump(nodes_dict, outfile)
        return res

    @staticmethod
    def get_contingency_table(cluster, unique_biome_list):
        terms_in_cluster = []
        for biome in cluster:
            terms_in_cluster.extend(biome.split(":"))

        res = {}
        for term, term_cnt in Counter(terms_in_cluster).items():
            if term_cnt <= 1:
                continue
            table = np.zeros((2, 2), dtype=int)
            table[0][0] = sum([1 for biome in cluster if term in biome])
            table[0][1] = len(cluster) - table[0][0]
            table[1][0] = sum([1 for biome in unique_biome_list if term in biome]) - table[0][0]
            table[1][1] = sum([1 for biome in unique_biome_list if term not in biome]) - (len(cluster) - table[0][0])
            res[term] = table
        return res

    def get_cluster_hover_info(self, nodes_dict_biome, unique_biome_list):
        # return res initialization
        cluster_ids, terms, uncorrected_pvalues, term_ratio = [], [], [], []
        cluster_hover_biome_count = {}
        cluster_hover_info = defaultdict(list)

        for cluster_id, cluster in nodes_dict_biome.items():
            tables = self.get_contingency_table(cluster, unique_biome_list)
            cluster_hover_biome_count[cluster_id] = len(cluster)
            for term, table in tables.items():
                oddsr, p = fisher_exact(table)
                cluster_ids.append(cluster_id)
                terms.append(term)
                uncorrected_pvalues.append(p)
                term_ratio.append("{:.1f}%".format((table[0][0] / (table[0][0] + table[1][0]) * 100)))
        # p value correction
        rejected, corrected_p = fdrcorrection(np.array(uncorrected_pvalues))
        np.set_printoptions(suppress=True)

        for i in range(len(terms)):
            cluster_hover_info[cluster_ids[i]].append("{}:{:.8f} ({})".format(terms[i], corrected_p[i], term_ratio[i]))
        return cluster_hover_info, cluster_hover_biome_count

    @staticmethod
    def my_pdist(biome_groupped):
        correlations_biome = biome_groupped.T.corr()
        dissimilarity = 1 - abs(correlations_biome)
        return squareform(dissimilarity)

    @staticmethod
    def my_linkage(dissimilarity):
        return linkage(dissimilarity, 'complete')

    def get_initial_tree(self):
        """
        Initialize a tree structure based on dendrogram using Plotly, at this step all hover info is "Hello World"
        :return: a figure
        :rtype: Plotly figure object
        """
        fig = ff.create_dendrogram(self.groupped_data_df,
                                   orientation='left',
                                   colorscale=[
                                       '#1f77b4',  # muted blue
                                       '#ff7f0e',  # safety orange
                                       '#2ca02c',  # cooked asparagus green
                                       '#d62728',  # brick red
                                       '#9467bd',  # muted purple
                                       '#8c564b',  # chestnut brown
                                       '#e377c2',  # raspberry yogurt pink
                                       '#7f7f7f',  # middle gray
                                       '#bcbd22',  # curry yellow-green
                                       '#17becf'   # blue-teal
                                   ],
                                   distfun=self.my_pdist,
                                   linkagefun=self.my_linkage,
                                   labels=self.groupped_data_df.index)

        fig.update_layout(xaxis=dict(
            tick0=0.5,
            dtick=0.75
        ))
        fig.update_layout(modebar_add="togglespikelines",
                          autosize=False,
                          width=1050,
                          height=1050,
                          paper_bgcolor="LightSteelBlue", )
        fig.update_xaxes(tickfont_size=8, tickangle=-45)
        # get axises tiles
        if self.abs:
            xaxis_title = "Distance: 1 - abs(correlation)"
        else:
            xaxis_title = "Distance: 1 - correlation"
        biome_cnt = len(self.groupped_data_df.index)
        yaxis_title = "Biome({})".format(biome_cnt)
        fig.update_layout(
            title="GO biome hierarchical clustering based on correlation",
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend_title="Legend Title",
            font=dict(
                size=8,
            )
        )
        fig.update_traces(hovertemplate='hello world')
        fig.update_traces(name='cluster', selector=dict(type='scatter'))
        return fig

    @staticmethod
    def update_hover_lables(fig_dict, cluster_hover_info, cluster_hover_biome_count):
        """
        This method updates initialized clustering tree with enrichment p value and biome count
        :param fig_dict: Plotly figure object in dictionary format
        :type fig_dict: dict
        :param cluster_hover_info: enrichment and p value
        :type cluster_hover_info: dict
        :param cluster_hover_biome_count: biome count in each cluster
        :type cluster_hover_biome_count: dict
        :return: updated figure
        :rtype: dict
        """
        fig_dict['data'].sort(key=lambda x:max(x['x']))
        for i in range(len((fig_dict['data']))):
            cluster_id = i + 1
            if cluster_id not in cluster_hover_info:
                fig_dict['data'][i]['hovertemplate'] = "No common term"
            else:
                sorted_p = sorted(cluster_hover_info[cluster_id], key=lambda x: x.split(':')[1].split(' ')[0])
                fig_dict['data'][i]['hovertemplate'] = "<br>".join(sorted_p)
            fig_dict['data'][i]['name'] = "cluster {}<br>{} biomes".format(cluster_id,
                                                                           cluster_hover_biome_count[cluster_id])
        return fig_dict

    def get_clustering_tree(self):
        """
        This method calls other methods to create the final clustering tree
        :return: None
        :rtype: None
        """
        # step 1: get Z
        Z = self.getZ()

        # step 2: get cluster info from Z, cluster id and all biomes it includes
        filtered_biome_list = self.biome_root_remover()
        cluster_info_dict = self.get_cluster_dict(Z, filtered_biome_list)

        # step 3: get enrichment and count for each cluster
        cluster_hover_info, cluster_hover_biome_count = self.get_cluster_hover_info(cluster_info_dict,
                                                                                    filtered_biome_list)

        # step 4: initialize a tree using Plotly
        fig = self.get_initial_tree()

        # step 5: update initialized tree with enrichment labels
        fig_dict = fig.to_dict()
        updated_fig_dict = self.update_hover_lables(fig_dict, cluster_hover_info, cluster_hover_biome_count)

        # step 6: save new fig
        pio.show(updated_fig_dict)
        pio.write_html(updated_fig_dict, os.path.join(self.output_path, 'GO_biome_hierarchical_clustering_tree.html'))
        pio.write_json(updated_fig_dict, os.path.join(self.output_path, 'GO_biome_hierarchical_clustering_tree.json'))