import logging
import os
import tempfile

import numpy as np
import sys

from mosaic.mcts import MCTS


class Search:
    """
        Main class to tune pipeline using Monte-Carlo Tree Search

        ...

        Attributes
        ----------
        mcts : class <mosaic.MCTS>
            object that run MCTS algorithm

        Methods
        -------
        run(nb_simulation=10, initial_configurations=[], nb_iter_to_generate_img=-1)
            Run nb_simulation of MCTS and initialize with initial_configurations
        """

    def __init__(self,
                 environment,
                 time_budget=3600,
                 seed=1,
                 policy_arg={},
                 exec_dir=None,
                 verbose=False):
        """Initialization algorithm.

        :param environment: environment class extending AbstractEnvironment
        :param time_budget: overall time budget
        :param seed: random seed
        :param policy_arg: specific option for MCTS policy
        :param exec_dir: directory to store tmp files
        """
        # config logger
        self.logger = logging.getLogger('mcts')
        self.logger.setLevel(logging.DEBUG)

        # execution directory
        if exec_dir is None:
            exec_dir = tempfile.mkdtemp()
        else:
            os.makedirs(exec_dir)

        hdlr = logging.FileHandler(os.path.join(exec_dir, "mcts.log"), mode='w')
        formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(funcName)s :: %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        if verbose:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        env = environment
        self.mcts = MCTS(env=env,
                         time_budget=time_budget,
                         policy_arg=policy_arg,
                         exec_dir=exec_dir)

        np.random.seed(seed)

    def run(self, nb_simulation=10, initial_configurations=[], nb_iter_to_generate_img=1, pb=None, status_txt=None, chart=None, status_txt2=None):
        """Run MCTS algorithm

        :param nb_simulation: number of MCTS simulation to run
        :param initial_configurations: path for generated image , optional
        :param nb_iter_to_generate_img: set of initial configuration, optional
        :return:
            configuration: best configuration found

        """
        self.logger.info("# Run {0} iterations of MCTS".format(nb_simulation))
        self.mcts.run(nb_simulation, initial_configurations, nb_iter_to_generate_img, pb, status_txt, chart, status_txt2)
        return self.mcts.bestconfig, self.mcts.bestscore
