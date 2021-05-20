import numpy as np
# import pynisher
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from mosaic.env import MosaicEnvironment


class Environment(MosaicEnvironment):

    def __init__(self, eval_func,
                 config_space,
                 mem_in_mb=3024,
                 cpu_time_in_s=300,
                 seed=42, data=None):
        super().__init__(seed)
        self.eval_func = eval_func
        self.config_space = config_space
        self.mem_in_mb = mem_in_mb
        self.cpu_time_in_s = cpu_time_in_s
        self.data = data

    def rollout(self, history=[]):
        while True:
            try:
                return self.config_space.sample_partial_configuration_with_default(history)
            except Exception:
                pass

    def next_move(self, history=[], info_children=[]):
        try:

            possible_params_ = list(
                self.config_space.get_possible_next_params(history))
            possible_params = self._can_be_selectioned(
                possible_params_, [v[0] for v in info_children], history)

            id_param = np.random.randint(0, len(possible_params))
            next_param = possible_params[id_param]

            next_param_cs = self.config_space._hyperparameters[next_param]

            if isinstance(next_param_cs, CategoricalHyperparameter):
                list_choice = next_param_cs.choices
            else:
                list_choice = [next_param_cs.sample(
                    self.rng) for _ in range(10)]

            value_param = np.random.choice(list_choice)

            history.append((next_param, value_param))
        except Exception as e:
            raise (e)

        possible_params.remove(next_param)
        is_terminal = len(possible_params) == 0
        return next_param, value_param, is_terminal

    def _can_be_selectioned(self, possible_params, child_info, history):
        history_ens = [v[0] for v in history]
        params_to_check = set(history_ens).intersection(set(possible_params))
        if len(params_to_check) > 0:
            for p in params_to_check:
                possible_params.remove(p)
                buffer = set()
                p_cs = self.config_space.get_hyperparameter(p)
                for new_val in p_cs.choices:
                    try:
                        self.config_space.sample_partial_configuration(
                            history + [(p, new_val)])
                        buffer.add(new_val)
                    except Exception:
                        pass

                if len([param for param in child_info if param == p]) < len(buffer):
                    possible_params.append(p)

        return possible_params

    def run_default_configuration(self):
        return self._evaluate(self.config_space.get_default_configuration())

    def get_nb_children(self, parameter, value, path):
        return 20

    def evaluate(self, config):

        # eval_func = pynisher.enforce_limits(
        #     mem_in_mb=self.mem_in_mb, cpu_time_in_s=self.cpu_time_in_s)(self.eval_func)
        # res = eval_func(config)
        res = self.eval_func(config, self.data)


        print(config, "score: ", res, end="\n\n")

        return res
