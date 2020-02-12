# -*- coding: utf-8 -*-
"""
This module loads settings from a configuration file.
"""

__author__ = 'ajshajib'

import yaml



class Config(object):
    """
    This class contains the methods to load an read YAML configuration
    files.
    """

    def __init__(self):
        pass

    @classmethod
    def load(cls, file):
        """
        Load configuration from `file`.
        :return:
        :rtype:
        """
        with open(file,'r') as f:
            settings = yaml.load(f)

        return settings


class ModelConfig(Config):
    """
    This class contains the methods to load and interact with modeling
    settings for a particular system.
    """

    def __init__(self, file):
        """
        Initiate a Model Config object from a given file.
        :param file: path to a settings file
        :type file: `string` or
        """
        super(ModelConfig, self).__init__()

        self.settings = self.load(file)

