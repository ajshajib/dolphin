# -*- coding: utf-8 -*-
"""This module provides classes to post process a model run output."""
__author__ = "ajshajib"

import numpy as np
import matplotlib.pyplot as plt

from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Sampling.parameters import Param
from lenstronomy.Util.class_creator import create_im_sim

from dolphin.processor import Processor
from dolphin.processor.config import ModelConfig


class Output(Processor):
    """This class lets."""

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
        """The output from the `lenstronomy...Fitting_sequence.fit_sequence()`.

        :return:
        :rtype:
        """
        if self._fit_output is None:
            raise ValueError(
                "Model output not specified!"
                "Load an output using the `load_output()`"
                "method!"
            )
        else:
            return self._fit_output

    @property
    def kwargs_result(self):
        """The `kwargs_result` after running a model by calling
        `lenstronomy...Fitting_sequence.fit_sequence()`.

        :return:
        :rtype:
        """
        if self._kwargs_result is None:
            raise ValueError(
                "Model output not specified!"
                "Load an output using the `load_output()`"
                "method!"
            )
        else:
            return self._kwargs_result

    @property
    def model_settings(self):
        """The `kwargs_result` after running a model by calling
        `lenstronomy...Fitting_sequence.fit_sequence()`.

        :return:
        :rtype:
        """
        if self._model_settings is None:
            raise ValueError(
                "Model output not specified!"
                "Load an output using the `load_output()`"
                "method!"
            )
        else:
            return self._model_settings

    @property
    def samples_mcmc(self):
        """The array of MCMC samples from the model run.

        :return:
        :rtype:
        """
        if self._samples_mcmc is None:
            return []
        else:
            return self._samples_mcmc

    @property
    def params_mcmc(self):
        """The non-linear parameters sampled with MCMC.

        :return:
        :rtype:
        """
        if self._params_mcmc is None:
            return []
        else:
            return self._params_mcmc

    @property
    def num_params_mcmc(self):
        """Number of sampled non-linear parameters in MCMC.

        :return:
        :rtype:
        """
        if self._params_mcmc is None:
            return 0
        else:
            return len(self._params_mcmc)

    def swim(self, *args, **kwargs):
        """Override the `swim` method of the `Processor` class to make it not
        callable."""
        raise NotImplementedError

    def load_output(self, lens_name, model_id):
        """Load output from file and save in class variables.

        :param lens_name: lens name
        :type lens_name: `str`
        :param model_id: model identifier provided at run initiation
        :type model_id: `str`
        :return: output dictionary
        :rtype: `dict`
        """
        output = self.file_system.load_output(lens_name, model_id)

        self._model_settings = output["settings"]
        self._kwargs_result = output["kwargs_result"]
        self._fit_output = output["fit_output"]
        self._multi_band_list_out = output["multi_band_list_out"]

        if self.fit_output[-1][0] == "emcee":
            self._samples_mcmc = self.fit_output[-1][1]
            self._params_mcmc = self.fit_output[-1][2]

        return output

    def get_model_plot(
        self,
        lens_name,
        model_id=None,
        kwargs_result=None,
        band_index=0,
        data_cmap="cubehelix",
    ):
        """Get the `ModelPlot` instance from lenstronomy for the lens.

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
        :return: `ModelPlot` instance, maximum pixel value of the image
        :rtype: `obj`, `float`
        """
        if model_id is None and kwargs_result is None:
            raise ValueError(
                "Either the `model_id` or the `kwargs_result` " "needs to be provided!"
            )

        if kwargs_result is None:
            self.load_output(lens_name, model_id)
            kwargs_result = self.kwargs_result

        multi_band_list_out = self._multi_band_list_out

        config = ModelConfig(
            lens_name=lens_name,
            io_directory=self.io_directory,
            settings=self.model_settings,
        )

        mask = config.get_masks()
        kwargs_model = config.get_kwargs_model()

        v_max = np.log10(multi_band_list_out[band_index][0]["image_data"].max())

        model_plot = ModelPlot(
            multi_band_list_out,
            kwargs_model,
            kwargs_result,
            arrow_size=0.02,
            cmap_string=data_cmap,
            image_likelihood_mask_list=mask,
            multi_band_type="multi-linear",
        )
        return model_plot, v_max

    def plot_model_overview(
        self,
        lens_name,
        model_id=None,
        kwargs_result=None,
        band_index=0,
        data_cmap="cubehelix",
        residual_cmap="RdBu_r",
        convergence_cmap="afmhot",
        magnification_cmap="viridis",
        v_min=None,
        v_max=None,
        source_v_min=None,
        source_v_max=None,
        print_results=False,
        show_source_light=False,
    ):
        """Plot the model, residual, reconstructed source, convergence, and
        magnification profiles. Either `model_id` or `kwargs_result` needs to be
        provided. `kwargs_result` is prioritized for plotting if both are provided.

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
        :type residual_cmap: `str` or `matplotlib.colors.Colormap`
        :param convergence_cmap: colormap for convergence plot
        :type convergence_cmap: `str` or `matplotlib.colors.Colormap`
        :param magnification_cmap: colormap for magnification plot
        :type magnification_cmap: `str` or `matplotlib.colors.Colormap`
        :param v_min: minimum plotting scale for the model, data, & source plot
        :type v_min: `float` or `int`
        :param v_max: maximum plotting scale for the model, data, & source plot
        :type v_max: `float` or `int`
        :param source_v_min: minimum plotting scale for the source plot
        :type source_v_min: `float` or `int`
        :param source_v_max: maximum plotting scale for the source plot
        :type source_v_max: `float` or `int`
        :param print_results: if true, prints the `kwargs_result` dictionary
        :type print_results: `bool`
        :param show_source_light: if true, replaces convergence plot with
            source light convolved lens decomposition plot and also replaces
            the magnification plot with the source-light subtracted data
            plot
        :type show_source_light: `bool`
        :return: `matplotlib.pyplot.figure` instance with the plots
        :rtype: `matplotlib.pyplot.figure`
        """
        if print_results:
            print_kwargs_result = kwargs_result
            if kwargs_result is None:
                print_kwargs_result = self.load_output(lens_name, model_id)[
                    "kwargs_result"
                ]
            print(print_kwargs_result)

        if v_max is None:
            model_plot, v_max = self.get_model_plot(
                lens_name,
                model_id=model_id,
                kwargs_result=kwargs_result,
                band_index=band_index,
                data_cmap=data_cmap,
            )
        else:
            model_plot = self.get_model_plot(
                lens_name,
                model_id=model_id,
                kwargs_result=kwargs_result,
                band_index=band_index,
                data_cmap=data_cmap,
            )[0]

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        model_plot.data_plot(
            ax=axes[0, 0], band_index=band_index, v_max=v_max, v_min=v_min
        )
        model_plot.model_plot(
            ax=axes[0, 1], band_index=band_index, v_max=v_max, v_min=v_min
        )
        model_plot.normalized_residual_plot(
            ax=axes[0, 2], band_index=band_index, cmap=residual_cmap, v_max=3, v_min=-3
        )
        model_plot.source_plot(
            ax=axes[1, 0],
            deltaPix_source=0.02,
            numPix=100,
            band_index=band_index,
            v_max=source_v_max,
            v_min=source_v_min,
            scale_size=0.4,
        )
        if not show_source_light:
            model_plot.convergence_plot(
                ax=axes[1, 1], band_index=band_index, cmap=convergence_cmap
            )
            model_plot.magnification_plot(
                ax=axes[1, 2], band_index=band_index, cmap=magnification_cmap
            )
        else:
            model_plot.subtract_from_data_plot(
                ax=axes[1, 1],
                band_index=band_index,
                lens_light_add=True,
                v_max=v_max,
                v_min=v_min,
            )
            model_plot.decomposition_plot(
                ax=axes[1, 2],
                text="Source light convolved",
                source_add=True,
                band_index=band_index,
                v_max=v_max,
                v_min=v_min,
            )
        fig.tight_layout()
        fig.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.05
        )

        return fig

    def plot_model_decomposition(
        self,
        lens_name,
        model_id=None,
        kwargs_result=None,
        band_index=0,
        data_cmap="cubehelix",
        v_min=None,
        v_max=None,
    ):
        """Plot lens light and source light model decomposition, both with convolved and
        unconvolved light. Either `model_id` or `kwargs_result` needs to be provided.
        `kwargs_result` is prioritized for plotting if both are provided.

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
        :param v_min: minimum plotting scale for the model, data, & source plot
        :type v_min: `float` or `int`
        :param v_max: maximum plotting scale for the model, data, & source plot
        :type v_max: `float` or `int`
        :return: `matplotlib.pyplot.figure` instance with the plots
        :rtype: `matplotlib.pyplot.figure`
        """

        if v_max is None:
            model_plot, v_max = self.get_model_plot(
                lens_name,
                model_id=model_id,
                kwargs_result=kwargs_result,
                band_index=band_index,
                data_cmap=data_cmap,
            )
        else:
            model_plot = self.get_model_plot(
                lens_name,
                model_id=model_id,
                kwargs_result=kwargs_result,
                band_index=band_index,
                data_cmap=data_cmap,
            )[0]

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        model_plot.decomposition_plot(
            ax=axes[0, 0],
            text="Lens light",
            lens_light_add=True,
            unconvolved=True,
            band_index=band_index,
            v_max=v_max,
            v_min=v_min,
        )
        model_plot.decomposition_plot(
            ax=axes[1, 0],
            text="Lens light convolved",
            lens_light_add=True,
            band_index=band_index,
            v_max=v_max,
            v_min=v_min,
        )
        model_plot.decomposition_plot(
            ax=axes[0, 1],
            text="Source light",
            source_add=True,
            unconvolved=True,
            band_index=band_index,
            v_max=v_max,
            v_min=v_min,
        )
        model_plot.decomposition_plot(
            ax=axes[1, 1],
            text="Source light convolved",
            source_add=True,
            band_index=band_index,
            v_max=v_max,
            v_min=v_min,
        )
        model_plot.decomposition_plot(
            ax=axes[0, 2],
            text="All components",
            source_add=True,
            lens_light_add=True,
            unconvolved=True,
            band_index=band_index,
            v_max=v_max,
            v_min=v_min,
        )
        model_plot.decomposition_plot(
            ax=axes[1, 2],
            text="All components convolved",
            source_add=True,
            lens_light_add=True,
            point_source_add=True,
            band_index=band_index,
            v_max=v_max,
            v_min=v_min,
        )
        fig.tight_layout()
        fig.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.05
        )

        return fig

    def get_reshaped_emcee_chain(
        self, lens_name, model_id, walker_ratio, burn_in=0, verbose=True
    ):
        """

        :param lens_name:
        :type lens_name:
        :param model_id:
        :type model_id:
        :param walker_ratio:
        :type walker_ratio:
        :param burn_in:
        :type burn_in:
        :param verbose:
        :type verbose:
        :return:
        :rtype:
        """
        self.load_output(lens_name, model_id)

        num_params = self.num_params_mcmc  # self.samples_mcmc.shape[1]
        num_walkers = walker_ratio * num_params
        num_step = int(len(self.samples_mcmc) / num_walkers)

        chain = np.empty((num_walkers, num_step, num_params))

        for i in np.arange(num_params):
            samples = self.samples_mcmc[:, i].T
            chain[:, :, i] = samples.reshape((num_step, num_walkers)).T
        if burn_in != 0:
            chain = chain[:, burn_in:, :]

        return chain

    def plot_mcmc_trace(
        self,
        lens_name,
        model_id,
        walker_ratio,
        burn_in=-100,
        verbose=True,
        fig_width=16,
        parameters_to_plot=[],
    ):
        """Plot the trace of MCMC walkers.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: model run identifier
        :type model_id: `str`
        :param walker_ratio: number of walkers per parameter in MCMC
        :type walker_ratio: `int`
        :param burn_in: number of burn-in steps to compute the medians after
            convergence of the MCMC chain
        :type: `int`
        :param verbose: if `True`, median values after burn-in will be printed
        :type verbose: `bool`
        :param fig_width: width of the figure
        :type fig_width: `float`
        :param parameters_to_plot: if not empty, list of parameters to plot
        :type fig_width: `list`
        :return: `matplotlib.pyplot.figure` instance with the plots
        :rtype: `matplotlib.pyplot.figure`
        """
        chain = self.get_reshaped_emcee_chain(
            lens_name, model_id, walker_ratio, burn_in=burn_in, verbose=verbose
        )

        num_params = self.num_params_mcmc
        num_step = chain.shape[1]

        if len(parameters_to_plot) == 0:
            parameter_list = np.arange(num_params)
        else:
            parameter_list = []
            for i in parameters_to_plot:
                if i in self.params_mcmc:
                    parameter_list.append(self.params_mcmc.index(i))
                else:
                    raise ValueError(
                        f"Parameter '{i}' not found. Available parameters: {self.params_mcmc}"
                    )

        mean_pos = np.zeros((num_params, num_step))
        median_pos = np.zeros((num_params, num_step))
        std_pos = np.zeros((num_params, num_step))
        q16_pos = np.zeros((num_params, num_step))
        q84_pos = np.zeros((num_params, num_step))

        # chain = np.empty((nwalker, nstep, ndim), dtype = np.double)
        for i in parameter_list:
            for j in np.arange(num_step):
                mean_pos[i][j] = np.mean(chain[:, j, i])
                median_pos[i][j] = np.median(chain[:, j, i])
                std_pos[i][j] = np.std(chain[:, j, i])
                q16_pos[i][j] = np.percentile(chain[:, j, i], 16.0)
                q84_pos[i][j] = np.percentile(chain[:, j, i], 84.0)

        fig, ax = plt.subplots(
            len(parameter_list),
            sharex="all",
            figsize=(fig_width, int(fig_width / 8) * len(parameter_list)),
        )
        last = num_step
        medians = []

        for n, i in enumerate(parameter_list):
            if verbose:
                print(
                    self.params_mcmc[i],
                    "{:.4f} ± {:.4f}".format(
                        median_pos[i][last - 1],
                        (q84_pos[i][last - 1] - q16_pos[i][last - 1]) / 2,
                    ),
                )
            if len(parameter_list) != 1:
                # ax[i].plot(mean_pos[i][:3000], c='b')
                ax[n].plot(median_pos[i][:last], c="g")
                # ax[i].axhline(np.mean(mean_pos[i][burnin:2900]), c='b')
                ax[n].axhline(np.median(median_pos[i][burn_in:last]), c="r", lw=1)
                ax[n].fill_between(
                    np.arange(last), q84_pos[i][:last], q16_pos[i][:last], alpha=0.4
                )
                # ax[i].fill_between(np.arange(last), mean_pos[i][:last] \
                # +std_pos[i][:last], mean_pos[i][:last]-std_pos[i][:last],
                # alpha=0.4)
                ax[n].set_ylabel(self.params_mcmc[i], fontsize=8)
                ax[n].set_xlim(0, last)
            else:
                ax.plot(median_pos[i][:last], c="g")
                ax.axhline(np.median(median_pos[i][burn_in:last]), c="r", lw=1)
                ax.fill_between(
                    np.arange(last), q84_pos[i][:last], q16_pos[i][:last], alpha=0.4
                )
                ax.set_ylabel(self.params_mcmc[i], fontsize=8)
                ax.set_xlim(0, last)

            medians.append(np.median(median_pos[i][burn_in:last]))

        return fig

    def get_param_class(self, lens_name, model_id):
        """Get `Param` instance for the lens model.

        :param lens_name: name of the lens
        :type lens_name: `str`
        :param model_id: model run identifier
        :type model_id: `str`
        :return: `Param` instance
        :rtype: `obj`
        """
        self.load_output(lens_name, model_id=model_id)

        config = ModelConfig(lens_name=lens_name, settings=self._model_settings)
        kwargs_params = config.get_kwargs_params()
        kwargs_model = config.get_kwargs_model()
        kwargs_constraints = config.get_kwargs_constraints()

        param = Param(
            kwargs_model,
            kwargs_params["lens_model"][2],
            kwargs_params["source_model"][2],
            kwargs_params["lens_light_model"][2],
            kwargs_params["point_source_model"][2],
            # kwargs_params['special'][2],
            # kwargs_params['extinction_model'][2],
            # kwargs_lens_init=kwargs_result['kwargs_lens'],
            kwargs_lens_init=kwargs_params["lens_model"][0],
            **kwargs_constraints,
        )
        return param

    def get_kwargs_from_args(
        self, lens_name, model_id, args, band_index=0, linear_solve=False, param=None
    ):
        """

        :param lens_name:
        :type lens_name:
        :param model_id:
        :type model_id:
        :return:
        :rtype:
        """
        if param is None:
            param = self.get_param_class(lens_name, model_id)

        kwargs = param.args2kwargs(args)

        if linear_solve:
            config = ModelConfig(lens_name=lens_name, settings=self._model_settings)

            # kwargs_numerics = config.get_kwargs_numerics()
            kwargs_model = config.get_kwargs_model()

            multi_band_list_out = self._multi_band_list_out

            # kwargs_data = multi_band_list_out[band_index][0]
            # kwargs_psf = multi_band_list_out[band_index][1]

            im_sim = create_im_sim(
                multi_band_list_out,
                "multi-linear",
                kwargs_model,
                bands_compute=None,
                image_likelihood_mask_list=(config.get_masks()),
                band_index=band_index,
            )

            im_sim.image_linear_solve(
                kwargs_lens=kwargs["kwargs_lens"],
                kwargs_source=kwargs["kwargs_source"],
                kwargs_lens_light=kwargs["kwargs_lens_light"],
                kwargs_ps=kwargs["kwargs_ps"],
                kwargs_extinction=kwargs["kwargs_extinction"],
                kwargs_special=kwargs["kwargs_special"],
                inv_bool=True,
            )

        return kwargs
