import os
import re
import pandas as pd
import numpy as np
import csv


def regex_filtered(a_list):
    """
    This method filters files name against certain pattern
    :param a_list: original list with items
    :type a_list: list
    :return: a new list after filtering
    :rtype: list
    """
    r = re.compile(".*\d_(GO|taxonomy)_abundances.*_v4.1.tsv")
    new_list = list(filter(r.match, a_list))
    return new_list


def collect_all_unique_terms(study_dict, studies):
    """
    This methods loops through the studies folder to collect all unique GO and taxonomy terms
    :param study_dict:
    :type study_dict: dictionary
    :param studies: path to studies folder
    :type studies: string
    :return: two lists of all unique GO and taxonomy terms
    :rtype: list, list
    """
    all_go_terms = set()
    all_taxa_terms = set()
    for study_name, files_list in study_dict.items():
        for file in files_list:
            tsv_path = os.path.join(studies, study_name, file)
            if re.match(r'.*GO.*', file):
                with open(tsv_path) as fd:
                    tsv_df = pd.read_csv(fd, sep='\t')
                    go_terms = set(tsv_df['GO'])
                    all_go_terms.update(go_terms)
            elif re.match(r'.*taxonomy.*', file):
                with open(tsv_path) as fd:
                    tsv_df = pd.read_csv(fd, sep='\t')
                    taxa_terms = set(tsv_df['#SampleID'])
                    all_taxa_terms.update(taxa_terms)
    return all_go_terms, all_taxa_terms




