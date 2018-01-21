import os

import click

import config


@click.command()
def main():
    for path in [
        config.SAVED_MODELS_PATH,
        config.PREDICTIONS_PATH,
        config.SUBMISSIONS_PATH,
    ]:
        os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    main()
