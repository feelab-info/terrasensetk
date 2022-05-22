import pandas as pd
import numpy as np
class IMetrics:
    """
        Base class for metric calculation
        Based on the work present in: https://github.com/ECGomes/nilm_metrics/blob/master/metrics/metrics_base.py
    """
    def __init__(self):
            return

    def check_metrics(self, results, metric_list,normalization_value=None):
        """
        :param results: Results Array
        :param metric_list: Set of metrics to calculate
        :param normalization_value: The ymax-ymin, not required
        :return: Dataframe containing metrics for the Results objects
        """

        # Go through a list of metrics to calculate
        unique_metrics_v1 = np.unique(np.array(metric_list))
        unique_metrics_v2 = self.__remove_undefinedMetrics__(unique_metrics_v1)

        # Calculate the metrics
        column_results = self.__calculate_metrics__(results, unique_metrics_v2,normalization_value)
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

    def __calculate_metrics__(self, results, metric_list,normalization_value=None):
        column_results = []
        for i in results:
            for j, result in enumerate(results[i]):
                
                metrics_results = {}
                metrics_results["algorithm"] = result.model.get_name() #get name of the algorithm
                metrics_results["run"] = j+1
                for calc in metric_list:
                    metrics_results[calc] = self.callFunction(calc,result.y_test, result.y_pred)
                    if(isinstance(metrics_results[calc],np.ndarray)):
                        metrics_results[calc] = metrics_results[calc][-1]
                    if normalization_value is not None:
                        metrics_results["n_"+calc] = metrics_results[calc]/normalization_value

                column_results.append(metrics_results)
        tmpdf = pd.DataFrame(column_results)
        #tmpdf.index.name = "Run"
        return tmpdf

    def __cleanup_results(self, results):
        return results