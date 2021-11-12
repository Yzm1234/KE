from sklearn.model_selection import train_test_split
import collections
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def dataset_split(df, train=0.7, val=0.2, test=0.1):
    df_train, df_rest = train_test_split(df, train_size=train, random_state=42, stratify=df['biome'])
    assert set(df_train['biome']) == set(df_rest['biome'])
    val_test_ratio = val/(val+test)
    df_val, df_test = train_test_split(df_rest, train_size=val_test_ratio, random_state=42, stratify=df_rest['biome'])
    assert set(df_train['biome']) == set(df_val['biome']) == set(df_test['biome'])
    return df_train, df_val, df_test


def dataset_balancing(df, each_class_count):
    X, y = df.iloc[:, 1:], df.iloc[:, 0]
    counter = collections.Counter(y).most_common()
    # create downsampling sampling_strategy map
    downsampler_dict = {}
    for label, count in counter:
        if count > each_class_count:
            downsampler_dict[label] = each_class_count
    # downsampling
    under_sampler = RandomUnderSampler(sampling_strategy=downsampler_dict)
    X_undersampled, y_undersampled = under_sampler.fit_resample(X, y)
    # upsampling
    over_sampler = RandomOverSampler(sampling_strategy='not majority')
    X_oversampled, y_oversampled = over_sampler.fit_resample(X_undersampled, y_undersampled)
    X_oversampled.insert(0, 'biome', y_oversampled)
    return X_oversampled

