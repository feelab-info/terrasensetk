
from performance.metrics.MetricsBase import MetricsBase
from sklearn import metrics
import pandas as pd
import numpy as np
import sklearn
from metric_utils import aux_error_checking
import math

class RegressionMetrics(MetricsBase):

    def cmd_rmse(self,gt,pred):
        """
        Calculates the root mean squared values
        """
        return math.sqrt(metrics.mean_squared_error(gt,pred.ravel()))

    def cmd_cv_rmsd(self, state_gt, state_pred):
        """
        Calculates the 1 - Covariance of the RMSD
        inv_cv = 1 - (RMSE / mean(ground_truth))
        """

        if aux_error_checking(state_gt, state_pred):
            return

        if np.mean(state_gt) == 0:
            return 0

        temp_cv_rmsd = 1 - (np.sqrt(sklearn.metrics.mean_squared_error(state_gt, state_pred)) / np.mean(state_gt))

        return temp_cv_rmsd