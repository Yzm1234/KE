import unittest
import filecmp
import aggregate as agg
import extract_biome as eb


class MyTestCase(unittest.TestCase):
    def test_generate_all_aggregated(self):
        study_dict = {'study1': ['ERP104187_GO_abundances_v4.1.tsv', 'ERP104187_taxonomy_abundances_SSU_v4.1.tsv'],
                      'study2': ['ERP104188_GO_abundances_v4.1.tsv', 'ERP104188_taxonomy_abundances_SSU_v4.1.tsv']}

        studies = 'data/studies'

        all_go_terms = ['go1', 'go2', 'go3', 'go4']
        all_taxa_terms = ['taxa1', 'taxa2', 'taxa3', 'taxa4']

        agg.generate_all_aggregated(study_dict, 'data/output/test.tsv', studies, all_go_terms, all_taxa_terms)
        self.assertTrue(filecmp.cmp('data/output/test.tsv', 'data/output/target.tsv'))

    def test_regex_filtered(self):
        my_list = ['biomes.json',
                   '.ipynb_checkpoints',
                   'analyses.json',
                   'ERP108758_GO_abundances_v4.1.tsv',
                   'ERP108758_GO_abundances_v5.0.tsv',
                   'samples.json',
                   'ERP108758_phylum_taxonomy_abundances_SSU_v4.1.tsv',
                   'ERP108758_phylum_taxonomy_abundances_SSU_v5.0.tsv',
                   'ERP108758_taxonomy_abundances_SSU_v4.1.tsv',
                   'ERP108758_taxonomy_abundances_SSU_v5.0.tsv',
                   'study.json']

        new_list = agg.regex_filtered(my_list)
        assert new_list == ['ERP108758_GO_abundances_v4.1.tsv', 'ERP108758_taxonomy_abundances_SSU_v4.1.tsv']

    def test_read_analyses_json(self):
        res_1 = eb.read_analyses_json("data/studies/", "study1", "run")
        res_2 = eb.read_analyses_json("data/studies/", "study2", "assembly")

        target_1 = {"SRRid2": {"sample_id": "SRS009922", "exptype": "assembly"},
                    "SRRid3": {"sample_id": "SRS009923", "exptype": "metagenomic"}}
        target_2 = {"ERZid7": {"sample_id": "SRS009927", "exptype": "undefined"},
                    "ERZid8": {"sample_id": "SRS009928", "exptype": "metagenomic"}}

        self.assertEqual(target_1, res_1)
        self.assertEqual(target_2, res_2)

    def test_read_samples_json(self):
        res_1 = eb.read_samples_json("data/studies/", "study1")
        res_2 = eb.read_samples_json("data/studies/", "study2")

        target_1 = {"SRS009922": "root:Environmental:Aquatic:Marine",
                    "SRS009923": "root:Environmental:Aquatic:Water"}
        target_2 = {"SRS009927": "root:Environmental:Aquatic:Marine",
                    "SRS009928": "root:Environmental:Aquatic",
                    "SRS009929": "root:Mixed"}

        self.assertEqual(target_1, res_1)
        self.assertEqual(target_2, res_2)

    def test_write_new_file(self):
        eb.write_new_file("data/studies/", 'data/output/target.tsv', 'data/output/test_write_new_file.tsv', 5)
        self.assertTrue(filecmp.cmp('data/output/test_write_new_file.tsv', 'data/output/target_2.tsv'))


if __name__ == '__main__':
    unittest.main()
