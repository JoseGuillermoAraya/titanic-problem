from commands.evaluate import evaluate
from commands.predict import predict
from commands.train import train
import click

@click.group()
def cli():
    pass

cli.add_command(evaluate)
cli.add_command(predict)
cli.add_command(train)

if __name__ == '__main__':
    cli()
