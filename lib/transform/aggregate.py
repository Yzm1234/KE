import os
import re
import pandas as pd
import numpy as np
import csv


def generate_all_aggregated(study_dict, output_file_path, studies, all_go_terms, all_taxa_terms):
    """
    This method generates aggregated
    :param study_dict:
    :type study_dict:
    :param output_file_path:
    :type output_file_path:
    :param studies:
    :type studies:
    :param all_go_terms:
    :type all_go_terms:
    :param all_taxa_terms:
    :type all_taxa_terms:
    :return:
    :rtype:
    """
    with open(output_file_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        # Write headers
        headers = ['id', 'study_id'] + list(all_go_terms) + list(all_taxa_terms)
        writer.writerow(headers)
        for study_name, files_list in study_dict.items():
            for file in files_list:
                tsv_path = os.path.join(studies, study_name, file)
                if re.match(r'.*GO.*', file):
                    with open(tsv_path) as fd:
                        go_df = pd.read_csv(fd, sep='\t')
                        go_ids = go_df.columns[3:]

                elif re.match(r'.*taxonomy.*', file):
                    with open(tsv_path) as fd:
                        taxa_df = pd.read_csv(fd, sep='\t')
                        taxa_ids = taxa_df.columns[1:]
            common_ids = list(np.intersect1d(go_ids, taxa_ids))

            if len(common_ids) == 0:
                print("\nstudy_name:", study_name)
                print("\n\tgo_ids:", go_ids)
                print("\n\ttaxa_ids:", taxa_ids)

            rows_dict = dict()
            # loop through go file
            for _, row in go_df.iterrows():
                go_term = row['GO']
                for _id in common_ids: # _id is run id or assembly id
                    if _id not in rows_dict:
                        rows_dict[_id] = {'id': _id, 'study_id': study_name}
                    rows_dict[_id][go_term] = row[_id]

            # loop through taxa file
            for _, row in taxa_df.iterrows():
                taxa_term = row['#SampleID']
                for _id in common_ids: # _id is run id or assembly id
                    if _id not in rows_dict:
                        raise ValueError("id %s is in taxa but not in go file" % _id)
                    rows_dict[_id][taxa_term] = row[_id]

            # write row_dict to output file
            for _, item in rows_dict.items():
                row_flat = [item['id'], item['study_id']]
                for go_term in all_go_terms:
                    row_flat.append(rows_dict[item['id']].get(go_term, 0))
                for taxa_term in all_taxa_terms:
                    row_flat.append(rows_dict[item['id']].get(taxa_term, 0))
                writer.writerow(row_flat)