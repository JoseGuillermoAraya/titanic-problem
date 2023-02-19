import click
import pandas as pd
import joblib
from ml_cli.utils.logger import get_logger
from ml_cli.utils.metrics import evaluate as metrics_evaluate
from ml_cli.data.make_dataset import make_dataset
from ml_cli.data.preprocess_data import preprocess_data

@click.command()
@click.option('--input-file', type=click.Path(exists=True), required=True, help='Path to the input CSV file')
@click.option('--model-file', type=click.Path(exists=True), required=True, help='Path to the trained model')
@click.option('--output-file', type=click.Path(), default='evaluation_results.csv', help='Path to the output CSV file')
@click.option('--log-file', default='evaluate.log', help='Path to the log file')
def evaluate(input_file, model_file, output_file, log_file):
    logger = get_logger(__name__, log_file)

    # Load the trained model
    logger.info(f'Loading model from {model_file}...')
    model = joblib.load(model_file)

    # Load the input data
    logger.info(f'Loading data from {input_file}...')
    X, y_true = make_dataset(input_file, 'Survived')

    # Preprocess the input data
    logger.info('Preprocessing data...')
    X = preprocess_data(X)

    # Make predictions and evaluate the model
    logger.info('Making predictions and evaluating model...')
    y_pred = model.predict(X)
    results = metrics_evaluate(y_true, y_pred)

    # Save the evaluation results to a CSV file
    logger.info(f'Saving evaluation results to {output_file}...')
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_file, index=False)

    click.echo(f'Successfully saved evaluation results to {output_file}.')

if __name__ == '__main__':
    evaluate()
