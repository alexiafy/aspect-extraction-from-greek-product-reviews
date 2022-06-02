from models.result_logs import ResultLogs
import time
from models.tagging_schemes import TaggingScheme
from datetime import datetime
from models.CRF.crf import CRFModel


def run_all_experiments():
    """
    Trains on all datasets with the 3 tagging systems and with and without POS tagging
    :return:
    """

    start_time = time.time()
    logs = ResultLogs()

    FOLDERNAME = "../../data/datasets/aspect_extraction_datasets/parts_1_2_3/"
    FILENAMES = ['ae_parts_1_2_3_usrnames_replaced']

    tagging_schemes = [TaggingScheme.IOB2]  # [TaggingSystem.IOB1, TaggingSystem.IOB2, TaggingSystem.BIOES]
    pos_tagging_list = [True]  # [True, False]

    total_experiments = len(FILENAMES) * len(tagging_schemes) * len(pos_tagging_list)
    experiment_idx = 0

    for file_name in FILENAMES:
        for IOB_tagging in tagging_schemes:
            for pos_tagging in pos_tagging_list:

                experiment_idx += 1

                print('[Running] File: \'' + file_name + '\', IOB tagging system: ' + IOB_tagging.value +
                      ', POS tagging included: ' + str(pos_tagging) + ', [' + str(experiment_idx) +
                      '/' + str(total_experiments) + ']')

                data_file_path = FOLDERNAME + file_name + ".csv"

                crf_instance = CRFModel(data_file_path, file_name, IOB_tagging, pos_tagging)
                crf_instance.train_model()
                # crf_instance.train_custom_model()
                logs.result_logs.append(crf_instance.logs)


    elapsed_time = time.time() - start_time
    logs.result_logs.append(['\n----------------------------',
                             'Total elapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
                             '----------------------------'])

    results_file_path = '../../results/CRF/parts_1_2_3/'
    results_file_name = 'crf_results_parts_1_2_3_(' + str(datetime.now().strftime('%d.%m.%Y_%H.%M.%S)')) + '.txt'

    logs.print_results_on_console()
    logs.results_to_csv(results_file_path, results_file_name)


def run_single_experiment():
    """
        Trains on a single dataset for experimentation
        :return:
    """

    FOLDERNAME = "../../datasets/aspect_extraction_datasets/parts_1_2/"
    FILENAME = 'aspect_extraction_data_part_1_2'
    data_file_path = FOLDERNAME + FILENAME + ".csv"

    logs = ResultLogs()

    crf_instance = CRFModel(data_file_path, FILENAME, TaggingScheme.IOB2, False)
    crf_instance.train_model()

    logs.result_logs.append(crf_instance.logs)
    logs.print_results_on_console()


    custom_sentences = ['Πηρα καφε nespresso, ηταν πολυ μετριος',
                        'Ο Λουμιδης απο τους καλυτερους καφεδες, ηπια χθες και πραγματικα τον απολαυσα',
                        'Όποια θέλει ναρθει να αράξουμε στο κρεβάτι όλη μέρα τρωγοντας Coco Pops, πίνοντας HEMO και βλεποντας Φρουτοπία, σήμερα μπορώ.',
                        'Ηπια νεσπρεσσο τρωγοντας Coco Pops, απο τα καλυτερα μου',
                        'Χθες δοκιμασα το korres πολυ ωραιο αρωμα και καλη ενυδατωση',
                        'Του korres εχω το ίδιο άρωμα σε γαλάκτωμα ειναι χειμωνιάτικο το παίρνω κάθε φορά που έρχεται φθινόπωρο ,μυρίζει σουπερ!!',
                        'RT @Christalakia:  Ειναι καποιο εθιμο που μου ξεφυγε? Ποιος πληρωνει τον ΟΤΕ τους? Γιατι να μην πληρωνουν μονοι τους τον ΟΤΕ τους? #εχω_αποριες',
                        'Και πηρα απο μασουτη Coco Pops η μεγαλυτερη απολαυση του κοσμου',
                        'πηγαιν Coco Pops περασα καλα',
                        'Δευτερη μερα @ COSMOTE που το μισο χωριο δεν εχει ιντερνετ!!']


    splitted_sent = [sub.split() for sub in custom_sentences]


    X_test = [crf_instance.sent2features(s) for s in splitted_sent]
    # print(X_test)
    pred = crf_instance.model.predict(X_test)

    for sent, pred in zip(splitted_sent, pred):
        print('\n')
        print(sent, '\n', pred)
        for word, label in zip(sent, pred):
            print(word, ': ', label)


if __name__ == '__main__':
    # run_single_experiment()
    run_all_experiments()

