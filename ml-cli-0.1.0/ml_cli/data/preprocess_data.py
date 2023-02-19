import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from ml_cli.utils.logger import get_logger

logger = get_logger(__name__)

def drop_features(df, features):
    """Drop features from the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        features (list): List of features to drop.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with dropped features.
    """
    logger.debug(f"Dropping features {features} from data")
    # Drop features
    df = df.drop(features, axis=1)
    
    return df

def impute_feature_with_mean_of_group(df, feature, group):
    """Impute feature with mean of that feature for a group
    of another feature in the data using Simple Imputer.

    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        feature (str): Feature to impute.
        group (str): Group to impute feature with mean of.

    Returns:
        df (pandas.DataFrame): Dataframe containing the data with imputed feature.
    """
    logger.debug(f"Imputing feature {feature} with mean of that feature for a group of {group} feature")
    # Impute feature with mean of that feature for a group of another feature
    imputer = SimpleImputer(strategy='mean')
    grouped = df.groupby(group)[feature].transform(lambda x: x.fillna(x.mean()))
    df[feature] = imputer.fit_transform(grouped.values.reshape(-1, 1))
    
    return df

def impute_missing_values(df, feature, strategy='mean'):
    """Impute missing values in the data using Simple Imputer.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        feature (str): Feature to impute.
        strategy (str): Strategy to impute missing values.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with imputed feature.
    """
    logger.debug(f"Imputing missing values in feature {feature} with strategy {strategy}")
    # Impute missing values
    imputer = SimpleImputer(strategy=strategy)
    df[feature] = imputer.fit_transform(df[feature].values.reshape(-1, 1))
    
    return df

def create_title_feature(df):
    """Create title feature. using create_feature_from_column.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with title feature.
    """
    logger.debug(f"Creating title feature from Name feature")
    # Create title feature
    df = create_feature_from_column(df, 'Name', 'Title', lambda x: x.split(',')[1].split('.')[0].strip())
    
    return df

def group_titles(df):
    """Group titles.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with grouped titles.
    """
    logger.debug(f"Grouping titles")
    # Group titles
    df['Title'] = df['Title'].replace(['Rev', 'Col', 'Jonkheer', 'Capt'], 'Other')
    df['Title'] = df['Title'].replace(['Dona', 'Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace(['Lady', 'Mme', 'the Countess'], 'Mrs')
    df['Title'] = df['Title'].replace(['Sir', 'Major', 'Dr', 'Don'], 'Mr')
    
    return df

def create_band_feature(df, feature, title, bins=5):
    """Create band feature. Encode the feature into a ordinal categorical feature with scikit learn
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        feature (str): Feature to create band feature from.
        title (str): Title of band feature.
        bins (int): Number of bins to create band feature from.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with band feature.
    """
    logger.debug(f"Creating band feature from {feature} feature")
    # Create band feature
    df[title] = pd.cut(df[feature].astype(int), bins)
    df[title] = OrdinalEncoder().fit_transform(df[title].values.reshape(-1, 1))

    return df

def create_feature_sum(df, features, title, const=0):
    """Create feature sum.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        features (list): List of features to sum.
        const (int): Constant to add to feature sum.
        title (str): Title of feature sum.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with feature sum.
    """
    logger.debug(f"Creating feature sum from {features} features")
    # Create feature sum
    df[title] = df[features].sum(axis=1) + const
    
    return df

def create_feature_from_column(df, feature, title, function):
    """Create feature from column.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        feature (str): Feature to create feature from.
        title (str): Title of feature.
        function (function): Function to apply to feature.

    Returns:
        df (pandas.DataFrame): Dataframe containing the data with feature.
    """
    logger.debug(f"Creating feature {title} from {feature} feature")
    # Create feature from column
    df[title] = df[feature].apply(function)
    
    return df

def encode_categorical_features(df, features):
    """One hot encode categorical features.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        features (list): List of features to encode.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with encoded categorical features.
    """
    logger.debug(f"Encoding categorical features {features}")
    encoder = OneHotEncoder(sparse_output=False)
    encoded_columns = encoder.fit_transform(df[features])
    new_columns = encoder.get_feature_names_out(features)
    encoded_df = pd.DataFrame(encoded_columns, columns=new_columns)
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(features, axis=1)

    return df

def preprocess_data(X):
    """Preprocess data.

    Args:
        X (pandas.DataFrame): Dataframe containing the features.

    Returns:
        X (pandas.DataFrame): Dataframe containing the features.
    """
    logger.debug(f"Preprocessing data")
    # Drop features
    features_to_drop = ['PassengerId', 'Cabin']
    X = drop_features(X, features_to_drop)

    # Create title feature
    X = create_title_feature(X)

    # Group titles
    X = group_titles(X)

    # Impute missing values
    X = impute_feature_with_mean_of_group(X, "Age", "Title")
    X = impute_missing_values(X, "Embarked", "most_frequent")

    # Feature engineering
    X = create_band_feature(X, "Age", "AgeBand", 10)
    X = create_feature_sum(X, ["SibSp", "Parch"], "FamilySize", 1)
    X = create_feature_from_column(X, "FamilySize", "IsAlone", lambda x: 1 if x == 1 else 0)
    X = create_band_feature(X, "Fare", "FareBand", 4)

    # Encode categorical features
    X = encode_categorical_features(X, ['Sex', 'Embarked', 'Title'])

    # Drop features
    X = drop_features(X, ['Name', 'Age', 'Fare', 'Ticket'])

    return X