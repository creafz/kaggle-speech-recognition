import os


BASE_DIR = '..'


def path(*rel_path):
    return os.path.join(BASE_DIR, *rel_path)


TRAIN_DIR_PATH = path('input', 'train', 'audio')
PICKLED_DATA_PATH = path('pickled_data')
SAVED_MODELS_PATH = path('saved_models')
PREDICTIONS_PATH = path('predictions')
SUBMISSIONS_PATH = path('submissions')
SAMPLE_SUBMISSION_PATH = path('input', 'sample_submission.csv')
TEST_DIR_PATH = path('input', 'test', 'audio')


SHUFFLE_SEED = 222
PROJECT_NAME = 'tensorflow_audio'
TENSORBOARD_LOGS_DIR = '/var/tensorboard_logs'
CUDNN_BENCHMARK = True
SEED = 42
NUM_WORKERS = 6
NUM_CLASSES = 12
AUDIO_SAMPLING_RATE = 16000
AUDIO_LENGTH = 16000
