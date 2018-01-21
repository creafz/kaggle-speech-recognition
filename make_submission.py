import os

import click
from click import option as opt, argument as arg
import numpy as np
import pandas as pd

import config
from dataset import idx_to_label
from utils import load_predictions


@click.command()
@arg('prediction-filenames', nargs=-1)
@opt('--submission-filename', type=str, required=True)
def main(
    prediction_filenames,
    submission_filename,
):
    predictions_list = []
    for filename in prediction_filenames:
        predictions_list.append(load_predictions(filename))
    predictions = np.vstack(predictions_list)
    mean_predictions = np.mean(predictions, axis=0)

    test_filenames = list(sorted(os.listdir(config.TEST_DIR_PATH)))
    predictions_by_file = {}
    for filename, file_predictions in zip(test_filenames, mean_predictions):
        idx = np.argmax(file_predictions)
        label = idx_to_label[idx]
        predictions_by_file[filename] = label

    sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    sample_submission.drop('label', axis=1, inplace=True)
    predictions_df = pd.DataFrame(
        list(predictions_by_file.items()),
        columns=['fname', 'label'],
    )
    submission_df = sample_submission.merge(predictions_df, on='fname')
    submission_df.to_csv(
        os.path.join(config.SUBMISSIONS_PATH, submission_filename),
        index=False,
    )


if __name__ == '__main__':
    main()
