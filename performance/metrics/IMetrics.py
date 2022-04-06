import pandas as pd
import numpy as np
class IMetrics:
    """
        Base class for metric calculation
        Based on the work present in: https://github.com/ECGomes/nilm_metrics/blob/master/metrics/metrics_base.py
    """
    def __init__(self):
            return

    def check_metrics(self, results, metric_list):
        """
        :param results: Results Array
        :param metric_list: Set of metrics to calculate
        :return: Dataframe containing metrics for the Results objects
        """

        # Go through a list of metrics to calculate
        unique_metrics_v1 = np.unique(np.array(metric_list))
        unique_metrics_v2 = self.__remove_undefinedMetrics__(unique_metrics_v1)

        # Calculate the metrics
        column_results = self.__calculate_metrics__(results, unique_metrics_v2)
        results = self.__cleanup_results(column_results)

        return results

    def checkFunction(self, name):
        fn = getattr(self, 'cmd_' + name, None)
        if fn is not None:
            return True
        else:
            print('Undefined metric call')
            return False

    def callFunction(self, name, gt, pred):
        fn = getattr(self, 'cmd_' + name, None)
        if fn is not None:
            return fn(gt, pred)
        else:
            print('Undefined metric call')
            return

    def callFunction(self, name, gt, pred):
        fn = getattr(self, 'cmd_' + name, None)
        if fn is not None:
            return fn(gt, pred)
        else:
            print('Undefined metric call')
            return

    def __remove_undefinedMetrics__(self, metric_list):
        new_list = metric_list.copy()
        for metric in metric_list:
            if self.checkFunction(metric):
                pass
            else:
                new_list = new_list[new_list != metric]
        return new_list

    def __calculate_metrics__(self, results, metric_list):
        column_results = {}
        for i,result in enumerate(results):
            metrics_results = {}
            for calc in metric_list:
                metrics_results[calc] = self.callFunction(calc,result.y_test, result.y_pred)
            metrics_results = pd.DataFrame(metrics_results)

            column_results[f"result {i}"] = metrics_results
        return column_results

    def __cleanup_results(self, results):
        return results