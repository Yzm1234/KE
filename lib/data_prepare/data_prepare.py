from sklearn.model_selection import train_test_split


def dataset_split(df, train=0.7, val=0.2, test=0.1):
    df_train, df_rest = train_test_split(df, train_size=train, random_state=42, stratify=df['biome'])
    assert set(df_train['biome']) == set(df_rest['biome'])
    val_test_ratio = val/(val+test)
    df_val, df_test = train_test_split(df_rest, train_size=val_test_ratio, random_state=42, stratify=df_rest['biome'])
    assert set(df_train['biome']) == set(df_val['biome']) == set(df_test['biome'])
    return df_train, df_val, df_test

