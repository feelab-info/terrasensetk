
from .IMetrics import IMetrics
from sklearn import metrics
import pandas as pd
import numpy as np
import sklearn
from .metric_utils import aux_error_checking
import math

class RegressionMetrics(IMetrics):
    def cmd_abse(self, state_gt, state_pred):
        """
        Calculate the Average Error of the ground truth vs predicted values
        ae = sum()
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_ae = 0
        temp_length = aux_get_size(state_gt)
        for i in np.arange(0, temp_length, 1):
            temp_ae += np.abs(state_gt[i] - state_pred[i])

        return temp_ae / temp_length

    def cmd_mae(self, state_gt, state_pred):
        """
        Calculates the Mean Absolute Error of the ground truth vs predicted values
        """

        if aux_error_checking(state_gt, state_pred):
            return

        return sklearn.metrics.mean_absolute_error(state_gt, state_pred)

    def cmd_ae(self, state_gt, state_pred):
        '''
        Calculate the Average Error of the ground truth vs predicted values
        ae = sum()
        '''

        if aux_error_checking(state_gt, state_pred):
            return

        temp_ae = 0
        temp_length = aux_get_size(state_gt)
        for i in np.arange(0, temp_length, 1):
            temp_ae += (state_pred[i] - state_gt[i])

        return temp_ae / temp_length

    def cmd_sde(self, state_gt, state_pred):
        """
        Calculate the standard deviation error
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_length = aux_get_size(state_gt)

        temp_placeholder = 0
        temp_delta_mean = self.cmd_ae(state_gt, state_pred)

        for i in np.arange(0, temp_length, 1):
            temp_placeholder += ((np.abs(state_pred[i] - state_gt[i]) - temp_delta_mean) ** 2)

        temp_sde = np.sqrt(temp_placeholder / temp_length)

        return temp_sde

    def cmd_rsquared(self, state_gt, state_pred):
        """
        Calculates the r-squared value
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_rsquared = sklearn.metrics.r2_score(state_gt, state_pred)

        return temp_rsquared

    def cmd_psde(self, state_gt, state_pred):
        """
        Percent Standard Deviation Explained: 1 - sqrt(1 - r-squared)
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_rsquared = self.cmd_rsquared(state_gt, state_pred)
        temp_psde = 1 - np.sqrt(1 - temp_rsquared)

        return temp_psde

    def cmd_ee(self, state_gt, state_pred):
        """
        Calculates the energy error of the predictions versus the ground truth values
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_length = aux_get_size(state_gt)
        temp_numerator = 0
        for i in np.arange(0, temp_length, 1):
            temp_numerator += np.abs(state_pred[i] - state_gt[i])

        temp_denominator = np.sum(state_gt)

        if temp_denominator == 0:
            return 0

        temp_ee = temp_numerator / temp_denominator

        return temp_ee

    def cmd_eav1(self, state_gt, state_pred, alpha_param=1.4):
        """
        Calculate energy accuracy
        """

        if aux_error_checking(state_gt, state_pred):
            return

        temp_ee = self.cmd_ee(state_gt, state_pred)
        temp_eav1 = math.exp(-alpha_param * temp_ee)

        return temp_eav1
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