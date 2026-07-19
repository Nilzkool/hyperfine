import pytest

from hyperfine import SearchRoutine, search_routine

FULL_SPACE = {
    'x': ['uniform', (-10, 10), False],
    'n': ['quniform', (1, 20, 1), False],
    'c': ['choice', ['a', 'b', 'c'], False],
    'r': ['randint', (10,), False],
    'lg': ['loguniform', (-3, 1), False],
    'qlg': ['qloguniform', (-3, 1, 0.5), False],
}

GRID_SPACE = {
    'x': ['uniform', (-10, 10), (20, 5)],
    'n': ['quniform', (1, 20, 1), (20, 4)],
    'c': ['choice', ['a', 'b', 'c'], False],
}


def objective(params):
    loss = 0.0
    if 'x' in params:
        loss += (params['x'] - 2.0) ** 2
    if 'n' in params:
        loss += (params['n'] - 10.0) ** 2 * 0.01
    if 'r' in params:
        loss += (params['r'] - 5.0) ** 2 * 0.01
    if 'lg' in params:
        loss += params['lg'] ** 2
    if 'qlg' in params:
        loss += params['qlg'] ** 2
    return {'loss': loss, 'status': 'ok'}


def test_search_routine_is_search_routine_alias():
    assert search_routine is SearchRoutine


def test_tpe_search_returns_well_formed_result_within_bounds():
    sr = SearchRoutine(obj_func=objective, space=FULL_SPACE, algo='tpe', max_evals=15, verbose=0)

    result = sr.search()

    assert set(result.keys()) == {'best_params', 'results'}
    assert 'loss' in result['results']

    bp = result['best_params']
    assert -10 <= bp['x'] <= 10
    assert 1 <= bp['n'] <= 20
    assert bp['c'] in ['a', 'b', 'c']
    assert 0 <= bp['r'] < 10
    import math
    assert math.exp(-3) <= bp['lg'] <= math.exp(1)
    # qloguniform rounds to the nearest multiple of q (0.5 here), so it can
    # legitimately round down to 0 or up slightly past the nominal upper bound.
    assert 0 <= bp['qlg'] <= math.exp(1) + 0.5


def test_random_search_returns_well_formed_result_within_bounds():
    sr = SearchRoutine(obj_func=objective, space=FULL_SPACE, algo='random', max_evals=15, verbose=0)

    result = sr.search()

    bp = result['best_params']
    assert -10 <= bp['x'] <= 10
    assert 1 <= bp['n'] <= 20
    assert bp['c'] in ['a', 'b', 'c']


def test_tpe_plus_grid_fine_tune_returns_well_formed_result():
    sr = SearchRoutine(obj_func=objective, space=GRID_SPACE, algo='tpe+grid', max_evals=10, verbose=0)

    result = sr.search()

    assert set(result.keys()) == {'best_params', 'results'}
    assert 'loss' in result['results']

    bp = result['best_params']
    assert isinstance(bp['x'], (int, float))
    assert isinstance(bp['n'], (int, float))
    assert bp['c'] in ['a', 'b', 'c']


def test_invalid_main_algo_returns_false():
    sr = SearchRoutine(obj_func=objective, space=FULL_SPACE, algo='invalid', max_evals=1, verbose=0)

    assert sr.search() is False
