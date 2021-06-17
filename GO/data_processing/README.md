# Data processing pipeline:<br />
![alt text](Mgnify_data_processing_flowchart.png "Title")
1. transform:from Mgnify database generating go_aggregated.tsv file<br />
2. get_labels: get biome labels for go_aggregated.tsv file<br />
3. remove_duplicated_rows: remove dplicated rows<br />
4. generate_dataset: creating dataset for Catboost model: the input tsv file will be randomly splited to train, validation and test dataset, this steps is not neccessary if pandas dataframe is used to run Catboost model <br />
5. feature_normalization: scale all numerical features into range (0,1)<br />
6. dataset_balancing: using oversample tool SMOTE to balance data<br />
