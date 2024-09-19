import pandas as pd

def clean_data(train, test, store):
    # Merge datasets, handle missing values, etc.
    train = pd.merge(train, store, on='Store', how='left')
    test = pd.merge(test, store, on='Store', how='left')

    # Fill missing values
    train['CompetitionDistance'].fillna(train['CompetitionDistance'].median(), inplace=True)
    train['PromoInterval'].fillna('None', inplace=True)

    return train, test
