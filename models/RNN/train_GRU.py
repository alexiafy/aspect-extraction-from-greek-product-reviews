import warnings
from GRU import *
import time
from datetime import datetime
import os

from models.tagging_schemes import TaggingScheme
from models.result_logs import ResultLogs


warnings.filterwarnings("ignore")


def run_single_experiment():

    logs = ResultLogs()
    start_time = time.time()
    start_datetime = datetime.now()

    RESULTS_FOLDER_PATH = '../../results/RNN/RNN_results_(' + str(start_datetime.strftime('%d.%m.%Y_%H.%M.%S')) + ')/'
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)  # succeeds even if directory exists.


    FOLDERNAME = "../../data/datasets/aspect_extraction_datasets/parts_1_2_3/"
    FILENAME = 'ae_parts_1_2_3_usrnames_replaced'


    # Run experiment
    # ---------------------------------------------------------
    model_parameters = {'EMBEDDING_SIZE': 100,
                        'Bidirectional': True,
                        'crf_layer': True,
                        'batch_size': 32,
                        'n_units': 128,
                        'dropout_rate': 0.4,
                        'epochs': 3}

    model = GRUClassifier(model_parameters, FILENAME, RESULTS_FOLDER_PATH, tagging_scheme=TaggingScheme.IOB2)
    model.load_data(FOLDERNAME)
    model.prepare_data()
    model.create_embeddings()

    model.define_model()
    model.fit_model()

    model.plot_acc_loss()
    model.evaluate()  # Print accuracy and loss
    # ---------------------------------------------------------

    logs.result_logs.append(model.logs)

    elapsed_time = time.time() - start_time
    logs.result_logs.append(['\n----------------------------',
                             'Elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
                             '----------------------------'])

    # --------------------------------------------
    results_file_name = 'results_(' + str(start_datetime.strftime('%d.%m.%Y_%H.%M.%S')) + ').txt'

    # logs.print_results_on_console()
    logs.results_to_csv(RESULTS_FOLDER_PATH, results_file_name)


def run_1st_experimentation():
    '''
    The first experimentation includes testing 4 files with different preprocessing steps
    and 3 different tag schemes (IOB1, IOB2, BIOES)
    :return:
    '''

    logs = ResultLogs()
    start_time = time.time()
    start_datetime = datetime.now()

    RESULTS_FOLDER_PATH = '../../results/RNN/RNN_results_(' + str(start_datetime.strftime('%d.%m.%Y_%H.%M.%S')) + ')/'
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)  # succeeds even if directory exists.

    # Define, train and evaluate model
    # --------------------------------------------

    FOLDERNAME = "../../data/datasets/aspect_extraction_datasets/parts_1_2_3/"

    FILENAMES = ['ae_parts_1_2_3',
                 'ae_parts_1_2_3_usrnames_replaced']

    tagging_schemes = [TaggingScheme.IOB1, TaggingScheme.IOB2, TaggingScheme.BIOES]

    for file_name in FILENAMES:
        for tagging_scheme in tagging_schemes:
            start_experiment_time = time.time()

            # Run experiment
            # ---------------------------------------------------------
            model_parameters = {'EMBEDDING_SIZE': 100,
                                'Bidirectional': True,
                                'crf_layer': False,
                                'batch_size': 32,
                                'n_units': 128,
                                'dropout_rate': 0.4}

            model = GRUClassifier(model_parameters, file_name, RESULTS_FOLDER_PATH, tagging_scheme=tagging_scheme)
            model.load_data(FOLDERNAME)
            model.prepare_data()
            model.create_embeddings()

            model.define_model()
            model.fit_model()

            model.plot_acc_loss()
            model.evaluate()  # Print accuracy and loss

            logs.result_logs.append(model.logs)

            elapsed_experiment_time = time.time() - start_experiment_time
            logs.result_logs.append(['\n----------------------------',
                                     'Elapsed time: ' + time.strftime("%H:%M:%S",
                                                                      time.gmtime(elapsed_experiment_time)),
                                     '----------------------------'])
        # --------------------------------------------

        # Calculate total elapsed time
        # --------------------------------------------
        elapsed_time = time.time() - start_time
        logs.result_logs.append(['\n----------------------------',
                                 'Total elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
                                 '----------------------------'])
        # --------------------------------------------
        results_file_name = 'results_(' + str(start_datetime.strftime('%d.%m.%Y_%H.%M.%S')) + ').txt'

        # logs.print_results_on_console()
        logs.results_to_csv(RESULTS_FOLDER_PATH, results_file_name)


def run_2nd_experimentation():
    '''
        The second experimentations focuses on testing
        LSTM vs Bi-lSTM vs GRU vs Bi-GRU
        one vs two layers of those
        with vs without additional Dense layer

        optimizers: Adam, RMSprop

        number of units: 64, 128, 256
        batch size: 32, 64
        dropout_rate: 0.3, 0.4, 0.5

        :return:
    '''

    logs = ResultLogs()
    start_time = time.time()
    start_datetime = datetime.now()

    RESULTS_FOLDER_PATH = '../../results/RNN/RNN_results_(' + str(start_datetime.strftime('%d.%m.%Y_%H.%M.%S')) + ')/'
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)  # succeeds even if directory exists.

    DATA_FOLDERNAME = "../../data/datasets/aspect_extraction_datasets/parts_1_2_3/"
    DATA_FILENAME = 'ae_parts_1_2_3_usrnames_replaced'

    # Run experiment
    # --------------------------------------------
    bidirectional_list = [True] #, False
    batch_size_list = [32, 64]
    n_units_list = [64, 128]
    dropout_rate_list = [0.2, 0.3, 0.4, 0.5]

    total_experiments = len(bidirectional_list) * len(n_units_list) * len(batch_size_list) * len(dropout_rate_list)
    experiment_idx = 0

    for bidirectional in bidirectional_list:
        for batch_size in batch_size_list:
            for n_units in n_units_list:
                for dropout_rate in dropout_rate_list:
                    start_experiment_time = time.time()
                    experiment_idx += 1
                    print('[Running] Experiment [' + str(experiment_idx) + '/' + str(total_experiments) + ']')

                    model_parameters = {'EMBEDDING_SIZE': 100,
                                        'Bidirectional': bidirectional,
                                        'crf_layer': True,
                                        'n_units': n_units,
                                        'batch_size': batch_size,
                                        'dropout_rate': dropout_rate,
                                        'epochs': 40}

                    model = GRUClassifier(model_parameters, DATA_FILENAME, RESULTS_FOLDER_PATH,
                                      tagging_scheme=TaggingScheme.IOB2)
                    model.load_data(DATA_FOLDERNAME)
                    model.prepare_data()
                    model.create_embeddings()

                    model.define_model()
                    model.fit_model()

                    model.plot_acc_loss()
                    model.evaluate()  # Print accuracy and loss

                    logs.result_logs.append(model.logs)

                    elapsed_experiment_time = time.time() - start_experiment_time
                    logs.result_logs.append(['\n----------------------------',
                                             'Elapsed time: ' + time.strftime("%H:%M:%S",
                                                                              time.gmtime(elapsed_experiment_time)),
                                             '----------------------------'])
    # --------------------------------------------

    # Calculate total elapsed time
    # --------------------------------------------
    elapsed_time = time.time() - start_time
    logs.result_logs.append(['\n----------------------------',
                             'Total elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
                             '----------------------------'])
    # --------------------------------------------

    results_file_name = 'results_(' + str(start_datetime.strftime('%d.%m.%Y_%H.%M.%S')) + ').txt'

    # logs.print_results_on_console()
    logs.results_to_csv(RESULTS_FOLDER_PATH, results_file_name)


if __name__ == '__main__':
    run_single_experiment()
    # run_1st_experimentation()
    # run_2nd_experimentation()





