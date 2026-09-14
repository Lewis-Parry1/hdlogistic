API Reference
=============

The public interface is organised around the statsmodels-compatible logistic
regression adapter and the state-equation solver used for high-dimensional
inference.

Top-level package exports
-------------------------

The main public classes and inference utilities can be imported directly from
``hdlogistic``:

.. code-block:: python

   from hdlogistic import (
       MDYPLLogistic,
       MDYPLLogisticResult,
       compute_sloe,
       compute_taus,
   )

.. automodule:: hdlogistic
   :members: compute_sloe, compute_taus

.. rubric:: Statsmodels adapter

.. automodule:: hdlogistic.adapters.statsmodels
   :members: MDYPLLogistic, MDYPLLogisticResult
   :show-inheritance:

State-equation solver
---------------------

.. automodule:: hdlogistic.solvers.state_equations_solver
   :members: SolverResult, StateParameters, ConvergenceCode
   :show-inheritance:

.. py:function:: solve_state_equation(kappa, signal_strength, alpha, start=None, **kwargs)

   Solve the three- or four-equation MDYPL state-evolution system.

   The solver returns a :class:`~hdlogistic.solvers.state_equations_solver.SolverResult`
   and a :class:`~hdlogistic.solvers.state_equations_solver.ConvergenceCode`.
   Use ``intercept`` in ``kwargs`` to solve the four-equation system.

Inference utilities
-------------------

The ``compute_taus`` and ``compute_sloe`` functions are documented in the
top-level package exports above and are available directly from ``hdlogistic``.
