# GO term abundance Catboost Classifier <br/>
This directory includes all notebooks for data processing, model training, result analyses.<br/>
## Purpose:<br/>
This repo is exploring the possibility to do prediction on biomes based on GO term abundance in bio samples. 
## Database:
Mgnify database
## Model:<br/>
Catboost multi-classifier model is selected.<br/>
Model parameters:
```
model = CatBoostClassifier(
    loss_function='MultiClass',
    custom_metric='Accuracy',
    learning_rate=0.15,
    random_seed=42,
    l2_leaf_reg=3,
    iterations=2000,
    task_type="GPU",
)
```  
## Data processing:<br/>
All Notebooks for data processing is held in directory `data_processing`<br/>
Processing workflow is show in the diagram below, dataset size is shown after each operation:
<img width="740" alt="Mgnify_data_processing_flowchart" src="https://user-images.githubusercontent.com/51136218/122092190-a41da100-cdd7-11eb-926f-3627c3c6a01a.png">

## Dataset:<br/>
1. Training dataset:
    - Scale: 62999 rows,  4403 columns, 90 biomes
    - Accuracy: 0.999
3. Validation dataset:
   - Scale: 18090 rows, 4403 columns, 90 biomes
   - Accuracy: 0.991
5. Test dataset:
   - Scale: 8911 rows, 4403 cilumns, 90 biomes
   - Accuracy: 0.989
## Result:<br/>
So far the hightest accuracy on test dataset is 0.989<br/>
Confusion matrix:<br/>

<img width="740" alt="Mgnify_data_processing_flowchart" src="https://user-images.githubusercontent.com/51136218/122092370-d7f8c680-cdd7-11eb-906c-b4d9df61fc74.png">

Feature importance on top features:<br/>

<img width="740" alt="Mgnify_data_processing_flowchart" src="https://user-images.githubusercontent.com/51136218/122094583-548ca480-cdda-11eb-8a14-6d74cd18e814.png">



