import unittest
import filecmp
import aggregate as agg
from . import regex_filtered


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

        new_list = regex_filtered(my_list)
        assert new_list == ['ERP108758_GO_abundances_v4.1.tsv', 'ERP108758_taxonomy_abundances_SSU_v4.1.tsv']


if __name__ == '__main__':
    unittest.main()
