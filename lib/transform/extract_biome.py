import csv
import json
import os
from tqdm.notebook import tqdm


def read_analyses_json(study, id_type):
    """"
    This method links assembly/run id to experiment type
    This function takes study_id and return run_id and sample_id matching pair in a hashmap.
    study_id: string: the id of a study
    return analyses_hash: analyses_hash: {id1: {'sample_id': sample_id_1, 'exptype': exptype_1},
                                          id2: {'sample_id': sample_id_2, 'exptype': exptype_2}}
    """
    studies_dir_path = "/global/cfs/cdirs/kbase/KE-Catboost/mgnify/data/mgnify/studies"
    analyses_path = os.path.join(studies_dir_path, study, 'analyses.json')
    analyses_hash = {}
    # read analyses.json
    with open(analyses_path) as f:
        analyses = json.load(f)
    # create analyses_hash
    for data in analyses['data']:
        # get run_id or assembly_id
        try:
            _id = data['relationships'][id_type]['data']['id']
        except KeyError as e:
            print("study: {} does not have {}_id in analyses.json. Error msg:{}".format(study, id_type, e))
            continue

        # get experiment type
        exp_type = 'undefined'
        try:
            exp_type = data['attributes']['experiment-type']
        except KeyError as e:
            continue

        # get sample id
        try:
            sample_id = data['relationships']['sample']['data']['id']
        except KeyError as e:
            print("study: {} does not have smaple_id in analyses.json. Error msg:{}".format(study, e))
            continue
        if _id and sample_id:
            analyses_hash[_id] = {"sample_id": sample_id, "exptype": exp_type}
    return analyses_hash


def read_samples_json(study):
    """
    Sample.json links sample id to its biome
    return: samples_hash = {sample_id1: biome1,
                            sample_id2: biome2}
    """
    studies_dir_path = "/global/cfs/cdirs/kbase/KE-Catboost/mgnify/data/mgnify/studies"
    samples_path = os.path.join(studies_dir_path, study, 'samples.json')
    samples_hash = {}
    # read sample.json
    with open(samples_path) as f:
        samples = json.load(f)
    # create sample_hash
    for data in samples['data']:
        biome = None
        try:
            sample_id = data['id']
        except KeyError as e:
            print("study: {} does not have sampoe_id in samples.json. Error msg:{}".format(study, e))
            continue
        try:
            biome = data['relationships']['biome']['data']['id']
        except KeyError as e:
            print("study: {} does not have biome in samples.json. Error msg:{}".format(study, e))
            continue
        if biome:
            samples_hash[sample_id] = biome
    return samples_hash


def write_new_file(input_file, output_file, row_count):
    with open(input_file, 'r') as f_in:
        with open(output_file, 'w') as f_out:
            reader = csv.reader(f_in, delimiter="\t")
            writer = csv.writer(f_out, delimiter="\t")
            header = next(reader)
            header.insert(2, 'sample_id')
            header.insert(3, 'biome')
            header.insert(4, 'exptype')
            writer.writerow(header)
            #         file_list = list(reader)
            study_hash = {}  # this dict is used to store the hash map for same study
            for i in tqdm(range(row_count)):
                row = next(reader)
                _id, study_id = row[0], row[1]
                if _id.startswith("SRR") or _id.startswith("ERR") or _id.startswith("ERZ"):
                    # determine the id type, run_id or assembly_id
                    if _id.startswith("SRR") or _id.startswith("ERR"):
                        id_type = 'run'
                    else:
                        id_type = 'assembly'

                    # if the stuty study_hash has been created, use it directly
                    if study_id in study_hash:
                        analyses_hash = study_hash[study_id]['analyses_hash']
                        samples_hash = study_hash[study_id]['samples_hash']
                    else:
                        analyses_hash = read_analyses_json(study_id, id_type)
                        samples_hash = read_samples_json(study_id)
                        study_hash[study_id] = {}
                        study_hash[study_id]['analyses_hash'] = analyses_hash
                        study_hash[study_id]['samples_hash'] = samples_hash

                    # try to extract the sample_id from analyses_hash, and write to new file
                    try:
                        sample_id = analyses_hash[_id]['sample_id']
                        exp_type = analyses_hash[_id]['exptype']
                        biome = samples_hash[sample_id]
                        row.insert(2, sample_id)
                        row.insert(3, biome)
                        row.insert(4, exp_type)
                        writer.writerow(row)
                    except KeyError:
                        print("{}_id({}) doesn't have sample id or biome".format(id_type, _id))
