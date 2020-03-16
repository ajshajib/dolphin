# -*- coding: utf-8 -*-
"""
This module provides classes to post process a model run output.
"""
__author__ = 'ajshajib'

import numpy as np

from dolphin.processor import Processor


class PostProcessor(Processor):
    """
    This class lets
    """
    def __init__(self, io_directory):
        """

        :param io_directory: path to the input/output directory. Should not end with slash.
        :type io_directory: `str`
        """
        super(PostProcessor, self).__init__(io_directory)

        self.fit_output = None
        self.kwargs_result = None
        self.model_settings = None
        self.samples_mcmc = None
        self.params_mcmc = None

    @property
    def num_params_mcmc(self):
        """
        Number of sampled non-linear parameters in MCMC.

        :return:
        :rtype:
        """
        if self.params_mcmc is not None:
            return len(self.params_mcmc)
        else:
            return 0

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

        self.model_settings = output['settings']
        self.kwargs_result = output['kwargs_result']
        self.fit_output = output['fit_output']

        if self.fit_output[-1][0] == 'EMCEE':
            self.samples_mcmc = self.fit_output[-1][1]
            self.params_mcmc = self.fit_output[-1][2]

    def thin_chain(self, walker_ratio, thin_factor=10,
                   burn_in=0):
        """
        Thin out a MCMC chain.

        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: model identifier provided at run initiation
        :type model_id: `str`
        :param walker_ratio: walker ratio per each parameter
        :type walker_ratio: `int`
        :param thin_factor: thinning factor
        :type thin_factor: `int`
        :param burn_in: number of burn-in steps to throw away
        :type burn_in: `int`
        :return: thinned chain with dim = (thinned n_steps*walker ratio, n_params)
        :rtype: `ndarray`
        """
        num_param = self.num_params_mcmc
        num_walker = walker_ratio * num_param
        num_step = int(self.samples_mcmc.shape[0] / num_walker)

        chain = np.empty((num_walker, num_step, num_param))

        for i in range(num_param):
            samples = self.samples_mcmc[:, i].T
            chain[:, :, i] = samples.reshape((num_step, num_walker)).T

        return chain[:, burn_in::thin_factor, :].reshape((-1, num_param))

    def save_thinned_chain(self, lens_name, model_id, walker_ratio,
                           thin_factor=10, burn_in=0):
        """
        Thin out a MCMC chain and save it to file.

        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: model identifier provided at run initiation
        :type model_id: `str`
        :param walker_ratio: walker ratio per each parameter
        :type walker_ratio: `int`
        :param thin_factor: thinning factor
        :type thin_factor: `int`
        :param burn_in: number of burn-in steps to throw away
        :type burn_in: `int`
        :return: `None`
        :rtype:
        """
        if self.samples_mcmc is None:
            self.load_output(lens_name, model_id)

        thinned_chain = self.thin_chain(walker_ratio, thin_factor, burn_in)

        output_dictionary = {
            'settings': self.model_settings,
            'kwargs_result': self.kwargs_result,
            'fit_output': [
                self.fit_output[:-1],
                ['EMCEE', thinned_chain, self.params_mcmc]
            ]
        }

        self.file_system.save_output(lens_name, '{}_thinned'.format(model_id),
                                     output_dictionary)

