burgers-equation-spectral-viscosity
====================================

Author: Jonah Miller (jonah.maxwell.miller@gmail.com)

# Description

We solve Burger's equation using spectral methods. Since there is a
shock present, a naive spectral method will not converge to the
correct entropy solution and will eventually crash. To resolve this,
we add the spectral viscosity first developed by Tadmor.

This amounts to adding a small amount of viscosity which only affects
higher-order spectral modes and converges away in the limit of
infinite resolution. This stabilizes the solution by draining energy
out of the system and, more importantly, ensures that the phase of the
shock is correct.

The convergence becomes linear at best, however, because the
discontinuity introduces Gibbs oscillations. This can be handled using
more advanced techniques such as Tadmor's adaptive mollifiers or the
Gegenbauer reconstruction. We do not pursue that here.

# Components

To see this in action, check out the
[IPython notebook.](burgers.ipynb). The spectral stencil is defined in
`orthopoly.py`.

# Sourcs

[1] Tadmor, Eitan. "Shock capturing by the spectral viscosity method." Computer Methods in Applied Mechanics and Engineering 80.1-3 (1990): 197-208.

[2] Gelb, Anne, and Eitan Tadmor. "Enhanced spectral viscosity approximations for conservation laws." Applied Numerical Mathematics 33.1-4 (2000): 3-21.

