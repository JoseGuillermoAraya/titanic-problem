import pandas as pd

def check_missing_values(df):
    """Check for missing values in a DataFrame and return a new DataFrame with the number and percentage of missing values for each column.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with missing values.
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_percent = round(df.isnull().sum() / df.shape[0] * 100, 2)
    df = pd.concat([missing_values, missing_percent], axis=1, keys=['Missing Values', 'Missing Percent'])
    df = df[df.iloc[:,1] != 0].sort_values('Missing Percent', ascending=False).round(1)
    return df

def check_data_types(df):
    """Check data types in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with data types.
    """
    # Check data types
    df = df.dtypes
    return df

def check_for_duplicates(df):
    """Check for duplicates in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with duplicates.
    """
    # Check for duplicates
    df = df.duplicated().sum()
    return df

def check_data_distribution(df):
    """Check data distribution in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with data distribution.
    """
    # Check data distribution
    df = df.describe()
    return df

def check_categorical_data_distribution(df):
    """Check categorical data distribution in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with categorical data distribution.
    """
    # Check categorical data distribution
    df = df.describe(include=['O'])
    return df

def check_data_skew_and_kurt(df):
    """Check column distribution, skewness, and kurtosis in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with column distribution.
    """
    # Check column distribution, skewness, and kurtosis
    skew = df.skew(numeric_only=True)
    kurtosis = df.kurtosis(numeric_only=True)
    df = pd.concat([skew, kurtosis], axis=1, keys=['Skew', 'Kurtosis'])
    return df

def check_for_outliers(df):
    """Check for outliers in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with outliers.
    """
    # Check for outliers
    df = df.boxplot()
    return df

def check_correlation(df):
    """Check correlation in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with correlation.
    """
    # Check correlation
    df = df.corr(numeric_only=True)
    return df

def average_and_count_column_by_feature(df, column, feature):
    """Average and count column by feature in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with average and count column by feature.
    """
    # Average and count column by feature
    df = df.groupby(feature)[column].agg(['mean', 'count'])
    return df

def check_feature_of_missing_column(df, column, feature):
    """Check feature of missing column in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with feature of missing column.
    """
    # Check feature of missing column
    df = df[df[column].isnull()][feature].value_counts()
    return df