from ml_cli.utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

def load_data(path):
    """Load data from path and return a pandas dataframe"""
    logger.debug(f"Loading data from {path}")
    df = pd.read_csv(path)
    return df

def split_data(df, target):
    """Split data into X and y"""
    logger.debug(f"Splitting data into X and y")
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y

def make_dataset(data_path, target):
    """Make dataset from data path and target"""
    logger.debug(f"Making dataset from {data_path} and target {target}")
    # Load data
    df = load_data(data_path)
    
    # Split data
    X, y = split_data(df, target)
    
    return X, y