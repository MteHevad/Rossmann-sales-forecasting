import pandas as pd

# Load raw data
train = pd.read_csv('../data/raw/train.csv', dtype={'StateHoliday': str})
store = pd.read_csv('../data/raw/store.csv')
test = pd.read_csv('../data/raw/test.csv', dtype={'StateHoliday': str})

# Merge store data with train and test datasets
train = pd.merge(train, store, on='Store', how='left')
test = pd.merge(test, store, on='Store', how='left')

# Handle missing values
train['CompetitionDistance'] = train['CompetitionDistance'].fillna(train['CompetitionDistance'].median())
train['Promo2SinceYear'] = train['Promo2SinceYear'].fillna(0)
train['Promo2SinceWeek'] = train['Promo2SinceWeek'].fillna(0)
train['PromoInterval'] = train['PromoInterval'].fillna('None')

# Feature engineering: date features
train['Date'] = pd.to_datetime(train['Date'])
train['Month'] = train['Date'].dt.month
train['DayOfWeek'] = train['Date'].dt.dayofweek

# Save the preprocessed data
train.to_csv('../data/processed/train_processed.csv', index=False)
test.to_csv('../data/processed/test_processed.csv', index=False)
