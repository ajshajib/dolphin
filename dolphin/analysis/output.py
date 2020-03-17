# -*- coding: utf-8 -*-
"""
This module provides classes to post process a model run output.
"""
__author__ = 'ajshajib'

import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.Plots.model_plot import ModelPlot

from dolphin.processor import Processor


class Output(Processor):
    """
    This class lets
    """
    def __init__(self, io_directory):
        """

        :param io_directory: path to the input/output directory. Should not
            end with slash.
        :type io_directory: `str`
        """
        super(Output, self).__init__(io_directory)

        self._fit_output = None
        self._kwargs_result = None
        self._model_settings = None
        self._samples_mcmc = None
        self._params_mcmc = None

    @property
    def fit_output(self):
        """
        The output from the `lenstronomy...Fitting_sequence.fit_sequence()`.

        :return:
        :rtype:
        """
        if self._fit_output is None:
            raise ValueError('Model output not specified!'
                             'Load an output using the `load_output()`'
                             'method!')
        else:
            return self._fit_output

    @property
    def kwargs_result(self):
        """
        The `kwargs_result` after running a model by calling
        `lenstronomy...Fitting_sequence.fit_sequence()`.

        :return:
        :rtype:
        """
        if self._kwargs_result is None:
            raise ValueError('Model output not specified!'
                             'Load an output using the `load_output()`'
                             'method!')
        else:
            return self._kwargs_result

    @property
    def model_settings(self):
        """
        The `kwargs_result` after running a model by calling
        `lenstronomy...Fitting_sequence.fit_sequence()`.

        :return:
        :rtype:
        """
        if self._model_settings is None:
            raise ValueError('Model output not specified!'
                             'Load an output using the `load_output()`'
                             'method!')
        else:
            return self._model_settings

    @property
    def samples_mcmc(self):
        """
        The array of MCMC samples from the model run.

        :return:
        :rtype:
        """
        if self._samples_mcmc is None:
            return []
        else:
            return self._samples_mcmc

    @property
    def params_mcmc(self):
        """
        The non-linear parameters sampled with MCMC.

        :return:
        :rtype:
        """
        if self._params_mcmc is None:
            return []
        else:
            return self._params_mcmc

    @property
    def num_params_mcmc(self):
        """
        Number of sampled non-linear parameters in MCMC.

        :return:
        :rtype:
        """
        if self._params_mcmc is None:
            return 0
        else:
            return len(self._params_mcmc)

    def swim(self, *args, **kwargs):
        """
        Override the `swim` method of the `Processor` class to make it
        not callable.
        """
        raise NotImplementedError

    def load_output(self, lens_name, model_id):
        """
        Load output from file and save in class variables.

        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: model identifier provided at run initiation
        :type model_id: `str`
        :return: None
        :rtype:
        """
        output = self.file_system.load_output(lens_name, model_id)

        self._model_settings = output['settings']
        self._kwargs_result = output['kwargs_result']
        self._fit_output = output['fit_output']

        if self.fit_output[-1][0] == 'EMCEE':
            self._samples_mcmc = self.fit_output[-1][1]
            self._params_mcmc = self.fit_output[-1][2]

        return output

    def plot_model_overview(self, lens_name, model_id=None,
                            kwargs_result=None, band_index=0,
                            data_cmap='cubehelix', residual_cmap='RdBu',
                            convergence_cmap='afmhot',
                            magnification_cmap='viridis'):
        """
        Plot the model, residual, reconstructed source, convergence,
        and magnification profiles. Either `model_id` or `kwargs_result`
        needs to be provided. `kwargs_result` is prioritized for plotting if
        both are provided.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: model run identifier
        :type model_id: `str`
        :param kwargs_result: lenstronomy `kwargs_result` dictionary. If
            provided, it will be used to plot the model, otherwise the model
            will be plotted from the saved/loaded outputs for `lens_name` and
            `model_id`.
        :type kwargs_result: `dict`
        :param band_index: index of band to plot for multi-band case
        :type band_index: `int`
        :param data_cmap: colormap for image, reconstruction, and source plots
        :type data_cmap: `str` or `matplotlib.colors.Colormap`
        :param residual_cmap: colormap for noise residual plot
        :type residual_cmap: str` or `matplotlib.colors.Colormap`
        :param convergence_cmap: colormap for convergence plot
        :type convergence_cmap: str` or `matplotlib.colors.Colormap`
        :param magnification_cmap: colormap for magnification plot
        :type magnification_cmap: str` or `matplotlib.colors.Colormap`
        :return: `matplotlib.pyplot.figure` instance with the plots
        :rtype: `matplotlib.pyplot.figure`
        """
        if model_id is None and kwargs_result is None:
            raise ValueError('Either the `model_id` or the `kwargs_result` '
                             'needs to be provided!')

        if kwargs_result is None:
            self.load_output(lens_name, model_id)
            kwargs_result = self.kwargs_result

        multi_band_list_out = self.get_kwargs_data_joint(
            lens_name)['multi_band_list']

        config = self.get_lens_config(lens_name)
        mask = config.get_masks()
        kwargs_model = config.get_kwargs_model()

        v_max = np.log10(
            multi_band_list_out[0][band_index]['image_data'].max())

        lens_plot = ModelPlot(multi_band_list_out, kwargs_model, kwargs_result,
                              arrow_size=0.02, cmap_string=data_cmap,
                              likelihood_mask_list=mask,
                              multi_band_type='multi-linear')

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        lens_plot.data_plot(ax=axes[0, 0], band_index=band_index, v_max=v_max)
        lens_plot.model_plot(ax=axes[0, 1], band_index=band_index, v_max=v_max)
        lens_plot.normalized_residual_plot(ax=axes[0, 2],
                                           band_index=band_index,
                                           cmap=residual_cmap, v_max=3,
                                           v_min=-3)
        lens_plot.source_plot(ax=axes[1, 0], deltaPix_source=0.02, numPix=100,
                              band_index=band_index, v_max=v_max)
        lens_plot.convergence_plot(ax=axes[1, 1], band_index=band_index,
                                   cmap=convergence_cmap)
        lens_plot.magnification_plot(ax=axes[1, 2],
                                     band_index=band_index,
                                     cmap=magnification_cmap)
        fig.tight_layout()
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0., hspace=0.05)

        return fig


