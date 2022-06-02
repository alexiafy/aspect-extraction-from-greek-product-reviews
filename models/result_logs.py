from datetime import datetime

class ResultLogs:

    def __init__(self):
        self.result_logs = []

    def results_to_csv(self, file_path, file_name):
        with open(file_path + file_name, 'w', encoding='UTF8', newline='') as file:
            for item in self.result_logs:
                for r in item:
                    file.write("%s\n" % r)

    def print_results_on_console(self):
        for item in self.result_logs:
            for r in item:
                print("%s" % r)

