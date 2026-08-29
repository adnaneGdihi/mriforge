.. _module_bloch:

Differentiable Bloch Simulation
===============================

**File Path:** ``src/infrastructure/physics/bloch_simulation.py``

High-Level Logic
----------------
This module provides a differentiable implementation of the Bloch equations, allowing the physics of MRI signal acquisition to be incorporated into the Deep Learning computational graph. It enables:

1.  **Physics-Informed Loss:** Gradients can flow from the signal error back to the tissue parameters (:math:`T_1, T_2`).
2.  **Forward Modeling:** Simulating acquisition artifacts given tissue maps.

Mathematical Core
-----------------
The module implements the **Ernst Equation** for Gradient Recalled Echo (GRE) sequences in the steady state.

.. math::
   S = M_0 \sin(\alpha) \frac{1 - E_1}{1 - \cos(\alpha)E_1} E_2

Where:
*   :math:`E_1 = \exp(-TR / T_1)`: Longitudinal relaxation.
*   :math:`E_2 = \exp(-TE / T_2^*)`: Transverse relaxation.
*   :math:`\alpha`: Flip angle.

This formulation is differentiable with respect to :math:`M_0, T_1, T_2`.

Class Breakdown
---------------

.. autoclass:: mriforge.infrastructure.physics.bloch_simulation.DifferentiableBlochSimulator
   :members:
   :undoc-members:
