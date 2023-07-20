import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def data_preprocessing(df):
    # standardise the temperature attribute in the dataset and convert Fahrenheit to Celsius
    df['Temperature'] = df['Temperature'].replace(to_replace=['\u00b0', 'C'], value='', regex=True)
    df['Temperature'] = df['Temperature'].replace(to_replace=['\u00b0', 'F'], value='', regex=True)
    df['Temperature'] = df['Temperature'].astype(float)
    df.loc[df['Factory'] == 'New York, U.S', 'Temperature'] = (df['Temperature'] - 32) * 5 / 9
    df.loc[df['Factory'] == 'Newton, China', 'Temperature'] = (df['Temperature'] - 32) * 5 / 9
    print('Data preprocessing and transformation ============================')
    print('Temperature features standardised to Degree Celsius.')

    # standardise the RPM attribute in the dataset to be positive
    df.loc[df['RPM'] < 0, 'RPM'] = df['RPM'] * -1
    print('RPM standardised.')

    # Split the Model attribute into Model and Year Produced.
    df[['Model', 'Year']] = df["Model"].apply(lambda x: pd.Series(str(x).split(",")))
    df['Year'] = df['Year'].astype(int)
    print('Production Year feature created.')

    # Remove duplicates of cars with same unique ID
    boolean_series = df.duplicated(keep='first')
    print('Number of duplicates to be removed:', df[boolean_series]['Car ID'].nunique())
    df.drop_duplicates(keep='first', inplace=True)

    #Dropping Car ID and Color features - no correlation with failure occurence
    df.drop(['Car ID', 'Color'], axis=1, inplace=True)
    print('Dropped columns - Car ID, Color')

    df['Condition'] = df['Failure A'] + df['Failure B'] + df['Failure C'] + df['Failure D'] + df['Failure E']
    print('Required binary feature created - 0=Non-Failure, 1=Failure')

    # Current number of rows in dataframe
    print('Current numbers of row in dataframe:', df.shape[0])
    print('Current numbers of column in dataframe:', df.shape[1],'\n')

    return df


def data_fitnorm(df):
    # Number of cars with failures
    Num_of_failed = df[df['Condition'] > 0]['Condition'].count()

    #Randomly picked 'Num_of_failed' samples out of non-failure samples
    df_0 = df[df['Condition'] == 0]
    df_0 = df_0.sample(n=Num_of_failed, replace=True)

    df.drop(df[df.Condition == 0].index, inplace=True)
    df = pd.concat([df_0, df], axis=0)

    # shuffle the DataFrame rows
    df = df.sample(frac=1)

    print('Shape of dataset for ML training:', df.shape, '\n')
    return df

def data_Xy(df):
    X = df[["Temperature", "RPM", "Fuel consumption", "Year", "Model", "Factory", "Usage", "Membership"]]
    # 0 = no failure detected/ 1 = failure detected
    y = df[['Condition']]

    print('Shape of X:', X.shape)
    print('Shape of y:', y.shape, '\n')

    return X,y

def data_transform(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    transformer = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                                  ("cat", categorical_transformer, categorical_features), ])

    return transformer
