import numpy as np
import pytest

from hyperfine import SearchRoutine


def _dummy_obj(params):
    return {'loss': 0.0, 'status': 'ok'}


def test_prepare_grid_fixed_and_quniform_params():
    space = {
        'c': ['choice', ['x', 'y', 'z'], False],
        'a': ['quniform', (1, 50, 1), (20, 5)],
    }
    sr = SearchRoutine(obj_func=_dummy_obj, space=space, algo='tpe', max_evals=1, verbose=0)

    flat_grid, param_names = sr._prepare_grid({'c': 1, 'a': 10})

    assert param_names == ['c', 'a']
    assert flat_grid.shape == (5, 2)
    # 'c' is not fine-tuned, so it stays fixed at its current (index) value
    assert np.all(flat_grid[:, 0] == 1)
    # 'a' is fine-tuned: radius=20%, N_points=5 around opt=10 -> [8, 9, 10, 11, 12]
    np.testing.assert_allclose(flat_grid[:, 1], [8, 9, 10, 11, 12])


def test_prepare_grid_randint_lb_equals_ub_epsilon_workaround():
    space = {
        'r': ['randint', (10,), (10, 5)],
    }
    sr = SearchRoutine(obj_func=_dummy_obj, space=space, algo='tpe', max_evals=1, verbose=0)

    # opt=1, radius=10% -> lb=ceil(0.9)=1, ub=floor(1.1)=1 -> lb==ub triggers the +1e-4 fix
    flat_grid, param_names = sr._prepare_grid({'r': 1})

    assert param_names == ['r']
    assert flat_grid.shape == (5, 1)
    assert np.all(flat_grid[:, 0] >= 1.0)
    assert np.all(flat_grid[:, 0] <= 1.0001)


def test_construct_grid_param_dict_roundtrips_choice_to_string():
    space = {
        'c': ['choice', ['x', 'y', 'z'], False],
        'a': ['quniform', (1, 50, 1), (20, 5)],
    }
    sr = SearchRoutine(obj_func=_dummy_obj, space=space, algo='tpe', max_evals=1, verbose=0)

    param_dict = sr._construct_grid_param_dict([1, 8], ['c', 'a'])

    assert param_dict == {'c': 'y', 'a': 8}


def test_construct_hyperopt_space_covers_all_sampling_types():
    space = {
        'x': ['uniform', (-10, 10), False],
        'n': ['quniform', (1, 20, 1), False],
        'c': ['choice', ['a', 'b', 'c'], False],
        'r': ['randint', (10,), False],
        'lg': ['loguniform', (-3, 1), False],
        'qlg': ['qloguniform', (-3, 1, 0.5), False],
    }
    sr = SearchRoutine(obj_func=_dummy_obj, space=space, algo='tpe', max_evals=1, verbose=0)

    hyperopt_space = sr._construct_hyperopt_space()

    assert set(hyperopt_space.keys()) == set(space.keys())
