import click
from ml_cli.data.make_dataset import make_dataset
from ml_cli.data.preprocess_data import preprocess_data
from ml_cli.models.xgb_model import TitanicXGBModel
from ml_cli.utils.logger  import get_logger

@click.command()
@click.option('--data-file', required=True, help='Path to the input data file')
@click.option('--log-file', default='train.log', help='Path to the log file')
@click.option('--model-file', default='model.pkl', help='Path to save the trained model')
@click.option('--max_depth', default=3, help='Maximum depth of a tree')
@click.option('--learning_rate', default=0.1, help='Learning rate')
@click.option('--objective', default='binary:logistic', help='Objective function')
@click.option('--eval_metric', default='logloss', help='Evaluation metric')
@click.option('--min_child_weight', default=1, help='Minimum sum of instance weight (hessian) needed in a child')
@click.option('--subsample', default=0.8, help='Subsample ratio of the training instances')
@click.option('--colsample_bytree', default=0.8, help='Subsample ratio of columns when constructing each tree')
@click.option('--min_child_weight', default=1, help='Minimum sum of instance weight (hessian) needed in a child')
@click.option('--num_boost_round', default=100, help='Number of boosting rounds')
@click.option('--early_stopping_rounds', default=10, help='Early stopping rounds')
@click.option('--seed', default=42, help='Random seed')

def train(data_file, log_file, model_file, **params):
    """
    Trains a XGB model on the input data.
    """
    logger = get_logger(__name__, log_file)
    # load and preprocess the input data
    logger.info('Loading and preprocessing data...')
    X, y = make_dataset(data_file, 'Survived')
    X = preprocess_data(X)

    # create a new instance of the model and train it on the data
    model = TitanicXGBModel(params)
    logger.info(f'Trining model for {params["num_boost_round"]} boosting rounds...')
    model.fit(X, y)

    # save the trained model to a file
    logger.info(f'Saving model to {model_file}...')
    model.save(model_file)

    logger.info('Training complete.')

if __name__ == '__main__':
    train()
