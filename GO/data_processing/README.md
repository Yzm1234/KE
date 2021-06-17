# Data processing pipeline:<br />
![alt text](Mgnify_data_processing_flowchart.png "Title")
1. **Aggregation**: Collect dataset from Mgnify database, aggregated samples from diffrent studies' directories
2. **Select and filtering**: 
    - Select rows and columns whose sum is not zero;<br />
    - Select samples with pipeline version 4.1; <br />
    - Select samples with exptype metagenomic or assembly
3. **Update biomes**: update "root:Mixed" biomes with more detailed labels <br />
    - e.g. "root:Mixed" -> "root:Mixed:temperate grassland:soil:Caterpillar"
4. **Feature normalization** (aka data scaling): using minmaxscaler to scale all numerical features in range [0,1] <br />
5. **Class balancing**: using [SMOTE](https://imbalanced-learn.org/stable/install.html) undersmapler downsampling majority classes and oversampler oversampling minority calsses
