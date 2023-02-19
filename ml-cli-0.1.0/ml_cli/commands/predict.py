import click
import joblib
import pandas as pd
from ml_cli.data.preprocess_data import preprocess_data
from ml_cli.utils.logger  import get_logger

# Define the command-line interface using Click
@click.command()
@click.option('--input_file', type=click.Path(exists=True), required=True, help='Path to the input CSV file')
@click.option('--output_file', type=click.Path(), default='predictions.csv', help='Path to the output CSV file')
@click.option('--model-file', default='model.bin', help='Path to the trained model')
@click.option('--log-file', default='predict.log', help='Path to the log file')
def predict(input_file, output_file, model_file, log_file):
    logger = get_logger(__name__, log_file)
    # Load the XGBoost model
    logger.info(f'Loading model from {model_file}...')
    model = joblib.load(model_file)

    # Load the input data
    logger.info(f'Loading data from {input_file}...')
    X = pd.read_csv(input_file)

    # Preprocess the data
    logger.info('Preprocessing data...')
    X = preprocess_data(X)

    # Make predictions using the XGBoost model
    logger.info('Making predictions...')
    y_pred = model.predict(X)

    # Save the predictions to a CSV file
    logger.info(f'Saving predictions to {output_file}...')
    output_data = pd.DataFrame({'predictions': y_pred})
    output_data.to_csv(output_file, index=False)

    click.echo(f'Successfully saved predictions to {output_file}')

if __name__ == '__main__':
    predict()
