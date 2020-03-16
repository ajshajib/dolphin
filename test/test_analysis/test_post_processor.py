# -*- coding: utf-8 -*-
"""
Tests for post_processor module.
"""
from pathlib import Path
import pytest
import numpy as np

from dolphin.processor import Processor
from dolphin.analysis.post_processor import PostProcessor

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / 'io_directory_example'


class TestProcessor(object):

    def setup_class(self):
        self.post_processor = PostProcessor(_TEST_IO_DIR)
        self.processor = Processor(_TEST_IO_DIR)

    @classmethod
    def teardown_class(cls):
        pass

    def test_swim(self):
        """
        Test that `swim` method is not accessible.
        :return:
        :rtype:
        """
        with pytest.raises(NotImplementedError):
            self.post_processor.swim()

    def test_num_params_mcmc(self):
        """
        Test `num_params_mcmc` property.
        :return:
        :rtype:
        """
        assert self.post_processor.num_params_mcmc == 0

        self.post_processor.params_mcmc = ['param1', 'param2']
        assert self.post_processor.num_params_mcmc == 2

    def test_load_output(self):
        """
        Test that outputs are saved and corresponding class variables
        are not None.

        :return:
        :rtype:
        """
        save_dict = {
            'settings': {'some': 'settings'},
            'kwargs_result': {'0': None, '1': 'str', '2': [3, 4]},
            'fit_output': [
                ['EMCEE',
                 [[2, 2], [3, 3]],
                 ['param1', 'param2']
                 ]
            ]
        }

        self.processor._save_output('test', 'post_process_load', save_dict)

        self.post_processor.load_output('test', 'post_process_load')

        assert self.post_processor.fit_output == save_dict['fit_output']
        assert self.post_processor.kwargs_result == save_dict['kwargs_result']
        assert self.post_processor.model_settings == save_dict['settings']

    def test_thin_chain(self):
        """
        Test `thin_chain` method. The chain has each param row filled with 0, 1,
        2, so on. The thinned chain should have the same parameter-wise mean
        after thinning.

        :return:
        :rtype:
        """
        num_param = 20
        walker_ratio = 10
        num_step = 500

        chain = np.ones((num_step * walker_ratio * num_param, num_param)) * \
                np.arange(num_param)

        self.post_processor.samples_mcmc = chain
        self.post_processor.params_mcmc = ['{}'.format(i) for i in
                                           range(num_param)]
        thinned_chain = self.post_processor.thin_chain(
            walker_ratio=walker_ratio, thin_factor=10, burn_in=200)

        assert np.all(np.mean(chain, axis=0) == np.mean(thinned_chain, axis=0))
        assert not np.all(np.mean(chain, axis=0) == np.mean(thinned_chain[:,
                                                            ::-1], axis=0))
        assert thinned_chain.shape == (int(((num_step - 200) * walker_ratio *
                                        num_param)/10), num_param)

    def test_save_thinned_chain(self):
        """

        :return:
        :rtype:
        """
        num_param = 10
        walker_ratio = 5
        num_step = 50

        chain = np.ones((num_step * walker_ratio * num_param, num_param))

        save_dict = {
            'settings': {'some': 'settings'},
            'kwargs_result': {'0': None, '1': 'str', '2': [3, 4]},
            'fit_output': [
                ['EMCEE',
                 chain,
                 ['{}'.format(i) for i in range(num_param)]
                 ]
            ]
        }
        self.processor._save_output('test', 'post_process_thin', save_dict)

        post_processor = PostProcessor(_TEST_IO_DIR)
        post_processor.save_thinned_chain('test', 'post_process_thin',
                                          walker_ratio=walker_ratio,
                                          thin_factor=5
                                          )

        post_processor.load_output('test', 'post_process_thin')

        assert post_processor.samples_mcmc.shape == (
            int(num_step * walker_ratio * num_param),
            num_param
        )
