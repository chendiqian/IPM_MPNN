from scipy.optimize._optimize import OptimizeResult
from scipy.optimize._linprog_util import (
    _parse_linprog, _presolve, _get_Abc, _LPProblem, _autoscale,
    _postsolve, _check_result)
from copy import deepcopy
from .linprog_ip import _linprog_ip


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
            bounds=None, method='interior-point', callback=None,
            options=None, x0=None, integrality=None):

    meth = method.lower()

    lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality)
    lp, solver_options = _parse_linprog(lp, options, meth)
    tol = solver_options.get('tol', 1e-9)

    iteration = 0
    complete = False  # will become True if solved in presolve
    undo = []

    # Keep the original arrays to calculate slack/residuals for original
    # problem.
    lp_o = deepcopy(lp)

    # Solve trivial problem, eliminate variables, tighten bounds, etc.
    rr_method = solver_options.pop('rr_method', None)  # need to pop these;
    rr = solver_options.pop('rr', True)  # they're not passed to methods
    c0 = 0  # we might get a constant term in the objective
    (lp, c0, x, undo, complete, status, message) = _presolve(lp, rr,
                                                             rr_method,
                                                             tol)
    assert not complete

    C, b_scale = 1, 1  # for trivial unscaling if autoscale is not used
    postsolve_args = (lp_o._replace(bounds=lp.bounds), undo, C, b_scale)

    if not complete:
        A, b, c, c0, x0 = _get_Abc(lp, c0)
        if solver_options.pop('autoscale', False):
            A, b, c, x0, C, b_scale = _autoscale(A, b, c, x0)
            postsolve_args = postsolve_args[:-2] + (C, b_scale)

        if meth == 'interior-point':
            x, status, message, iteration, callback_outputs = _linprog_ip(
                c, c0=c0, A=A, b=b, callback=callback,
                postsolve_args=postsolve_args, **solver_options)

    # Eliminate artificial variables, re-introduce presolved variables, etc.

    x, fun, slack, con = _postsolve(x, postsolve_args, complete)

    status, message = _check_result(x, fun, status, slack, con, lp_o.bounds, tol, message, False)


    sol = {
        'x': x,
        'fun': fun,
        'slack': slack,
        'con': con,
        'status': status,
        'message': message,
        'nit': iteration,
        'success': status == 0,
        'intermediate': callback_outputs}

    return OptimizeResult(sol)
