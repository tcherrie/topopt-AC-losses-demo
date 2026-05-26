"""Optimization utilities

Provide utilities to optimize the electrical machine.

Functions defined here:
- project               (optimization algorithm)
- gradient_descent      (optimization algorithm)
- taylor_test           (gradient check)
- plot_taylor_test      (gradient check)
- solve_adjoint         (gradient computation)
- dd_joule_losses       (gradient computation)
- partiald_joule_losses (gradient computation)
- d_joule_losses        (gradient computation)


A Large Language Model (GPT-5.5 from Open AI, free version) was used to help with the code and generate
the docstrings of the functions. The authors have written the initial code, carefully checked and post-edited
the content of this file, and take full responsability of its content.
This software is provided "as is" without warranty of any kind, and can be used, shared ad modified under the terms of GNU LGPL license.
"""

#%% Metadata

__author__ = "Théodore Cherrière"
__copyright__ = "Copyright 2026, CentraleSupélec, SAFRAN"
__credits__ = ["Théodore Cherrière", "Alexis Pons", "Guillaume Krebs",
                    "Adrien Mercier", "Loucif Benmamas", "Sulivan Küttler"]
__license__ = "GNU LGPL"
__version__ = "0.1"
__maintainer__ = "Théodore Cherrière"
__email__ = "theodore.cherriere@centralesupelec.fr"
__status__ = "Development"

#%% Import

import matplotlib.pyplot as plt
import numpy as np
import ngsolve as ngs

from ngsolve.webgui import Draw
from copy import copy
from utils.physics import current_density
from utils.physics import electric_field

#%% Projected gradient descent

def project(x:      ngs.GridFunction,
            x_min:  float,
            x_max:  float
            ) -> ngs.GridFunction:
    """
    Clamp a ngs.GridFunction vector between lower and upper bounds.

    This function applies element-wise projection (clamping) to the degrees
    of freedom of a NGSolve ngs.GridFunction, ensuring that all values remain
    within the interval [x_min, x_max].

    Parameters
    ----------
    x : ngs.GridFunction
        Input finite element field to be clamped in-place.

    x_min : float
        Lower bound applied to all entries.

    x_max : float
        Upper bound applied to all entries.

    Returns
    -------
    ngs.GridFunction
        The same ngs.GridFunction instance with its values modified in-place.
    """

    # Clamp values between x_min and x_max
    x.vec.data = np.clip(x.vec, x_min, x_max)

    return x


def gradient_descent(state :        callable,           # takes x0 ngs.GridFunction as input, returns "results" dictionnary
                     objective :    callable,           # takes "results" dictionnary as input, returns float
                     d_objective :  callable,           # takes "results" dictionnary as input, returns Gridfunction
                     x0 :           ngs.GridFunction, 
                     descent :      callable    = lambda x: -x, # takes d_objective ngs.GridFunction as input, returns descent ngs.GridFunction
                     step :         float       = 1,
                     x_min :        float       =  -np.inf, 
                     x_max :        float       =   np.inf,
                     maxit :        int         = 50,
                     tol :          float       = 1e-6,
                     step_min :     float       = 1e-2,
                     fac_increase_step : float  = 1.2,
                     fac_decrease_step : float  = 0.5,
                     draw :         bool = True,
                     verbose :      int = 2
                     )-> dict:
    """
    Constrained gradient descent optimizer for NGSolve ngs.GridFunctions.

    This routine performs a basic adaptive gradient descent on a
    finite-element field. It supports box constraints via projection,
    adaptive step size control, and optional real-time visualization.

    The algorithm evaluates a physical state, computes an objective function,
    and updates the design variable using a provided descent direction.

    Parameters
    ----------
    state : callable
        Function mapping a ngs.GridFunction to a simulation result dictionary.

    objective : callable
        Function mapping the result dictionary to a scalar objective value.

    d_objective : callable
        Function mapping the result dictionary to the gradient
        (as a ngs.GridFunction).

    x0 : ngs.GridFunction
        Initial design variable.

    descent : callable, optional
        Mapping from gradient to descent direction.
        Default is -gradient.

    step : float, optional
        Initial step size.

    x_min : float, optional
        Lower bound for projection constraint.

    x_max : float, optional
        Upper bound for projection constraint.

    maxit : int, optional
        Maximum number of iterations.

    tol : float, optional
        Convergence tolerance on the normalized criterion.

    step_min : float, optional
        Minimum allowable step size before stopping.

    fac_increase_step : float, optional
        Multiplicative factor to increase step size after successful iteration.

    fac_decrease_step : float, optional
        Multiplicative factor to decrease step size after rejection.

    draw : bool, optional
        If True, displays and updates a live visualization.

    verbose : int, optional
        Verbosity level (0 = silent, 1 = minimal, 2 = detailed).

    Returns
    -------
    dict
        Dictionary containing optimization history:
        - "solution": list of ngs.GridFunction iterates
        - "objective": list of objective values
        - "gradient": list of gradients
        - "criterion": convergence indicator history
        - "state": last computed state
    """
    
    # Initialization
    x_list = [x0]
    x_accepted = x0
    F_list = []
    dF_list = []
    crit_list = [1]
    
    characteristic_function = ngs.GridFunction(x0.space)
    characteristic_function.Set(1)
    surface = ngs.Integrate(characteristic_function, x0.space.mesh)

    if draw:
        scene = Draw(x_list[-1], x0.space.mesh,
                         min = x_min, max = x_max,
                         settings = {"Objects" : {"Wireframe" : False}, 
                                     "Colormap" : {"ncolors" : 32}})
    
    iter = 0

    while True:
        iter += 1

        # 1) Compute physical state
        result = state(x_list[-1])
    
        # 2) Compute objective function
        F_list.append(objective(result))

        # 3) Update step & compute stop criterion
        if len(F_list) == 1:            # initial descent
            dF_list.append(d_objective(result))
            x_accepted, F_accepted, dF_accepted = x_list[-1], F_list[-1], dF_list[-1]
            dx_crit = ngs.GridFunction(x0.space)
            dx_crit.Set(x_list[-1] - step_min * dF_list[-1])
            crit_ref = ngs.Integrate(ngs.Norm(project(dx_crit, x_min, x_max) - x_list[-1]), x0.space.mesh) / (step_min*surface)
            if verbose >= 2: print(f"it {iter} ✅| obj = {F_list[-1] :.3e} | {step = :.2e} | crit = {crit_list[-1] :.2e}")
            
        elif F_list[-1] < F_accepted:   # accept the step
            dF_list.append(d_objective(result))
            x_accepted, F_accepted, dF_accepted = x_list[-1], F_list[-1], dF_list[-1]
            dx_crit.Set(x_list[-1] - step_min * dF_list[-1])
            crit_list.append(ngs.Integrate(ngs.Norm(project(dx_crit, x_min, x_max) - x_list[-1]), x0.space.mesh) / (step_min*surface) / crit_ref)
            step *= fac_increase_step
            if verbose >= 2: print(f"it {iter} ✅| obj = {F_list[-1] :.3e} | {step = :.2e} | crit = {crit_list[-1] :.2e}")

        else:                           # reject the step
            step *= fac_decrease_step
            crit_list.append(crit_list[-1])
            if verbose >= 2:  print(f"it {iter} ❌| obj = {F_list[-1] :.3e} | {step = :.2e} | crit = {crit_list[-1] :.2e}")

        # 4) Check convergence
        if crit_list[-1] < tol: # stop if converged
            print(f"Converged! {crit_list[-1] = :.2e} < {tol :.2e}")
            break

        if iter >= maxit:
            print("Maximum number of iterations reached!")
            break

        if step < step_min:
            print("Step lower than minimum step size!")
            break

        # 5) Update variable
        x_test = copy(x_accepted)
        x_test.vec.data += step * descent(dF_accepted).vec
        x_list.append(project(x_test, x_min, x_max))

        if draw:
            scene.Redraw(x_list[-1], x0.space.mesh,
                         min = x_min, max = x_max,
                         settings = {"Objects" : {"Wireframe" : False}, 
                                     "Colormap" : {"ncolors" : 32}})
            
    return {"solution" :  x_list,
            "objective" : F_list,
            "gradient" :  dF_list,
            "criterion" : crit_list,
            "state" : result}


#%% Taylor test

def taylor_test(F:      callable, 
               dF:      callable, 
               x0:      ngs.GridFunction, 
               pert:    ngs.GridFunction    = None, 
               H:       list|np.ndarray   = np.logspace(-9,1,20)
               ) -> tuple:
    """
    Perform a Taylor test to verify correctness of a functional gradient.

    This function checks whether the provided derivative `dF` is consistent
    with the finite difference behavior of the functional `F`. It evaluates
    the first-order Taylor expansion error for a range of step sizes.

    Parameters
    ----------
    F : callable
        Functional mapping a ngs.GridFunction to a scalar value.

    dF : callable
        Derivative of F, returning a ngs.GridFunction representing the gradient of F.

    x0 : ngs.GridFunction
        Base point at which the Taylor expansion is evaluated.

    pert : ngs.GridFunction, optional
        Perturbation direction. If None, a random perturbation is generated.

    H : array-like, optional
        Array of step sizes used for the Taylor expansion test.

    Returns
    -------
    H : array-like
        Step sizes used in the test.

    taylor_remainder : list
        Absolute Taylor remainder |F(x+h p) - F(x) - h dF(x)·p|.

    dF0 : float
        Directional derivative computed from dF at x0.

    dF_estimated : list
        Finite difference approximation of the directional derivative.

    Notes
    -----
    - A correct gradient should yield a remainder scaling like O(h²).
    - The estimated derivative should converge to dF0 as h → 0.
    """

    # Evaluate baseline functional value
    F0 = F(x0)

    # Generate random perturbation if none is provided
    if pert is None:
        pert = ngs.GridFunction(x0.space)
        pert.vec.data = np.random.rand(len(pert.vec)) * x0.vec

    # Directional derivative using provided gradient
    dF0 = ngs.InnerProduct(dF(x0).vec, pert.vec)

    taylor_remainder = []
    dF_estimated = []

    # Loop over step sizes
    for h in H:
        # Perturbed state
        x1 = x0 + h * pert

        # Finite difference derivative approximation
        dF_estimated.append((F(x1) - F0) / h)

        # Taylor remainder (should scale ~ h^2 if correct)
        taylor_remainder.append(abs(F(x1) - F0 - dF0 * h))

    return H, taylor_remainder, dF0, dF_estimated


def plot_taylor_test(H :                np.ndarray, 
                     taylor_remainder : np.ndarray
                     ) -> None:
    """
    Plot the Taylor test convergence behavior.

    This function visualizes the Taylor remainder as a function of step size
    on a log-log scale and compares it against an O(h^2) reference slope.

    Parameters
    ----------
    H : array-like
        Step sizes used in the Taylor test.

    taylor_remainder : array-like
        Absolute Taylor remainder values corresponding to each step size.

    Returns
    -------
    None
        Displays a matplotlib log-log plot.

    Notes
    -----
    - A correctly implemented gradient should produce a curve parallel to h^2.
    """

    # Plot observed Taylor remainder
    plt.loglog(H, taylor_remainder, 'o')

    # Reference slope O(h^2)
    plt.loglog(H, H**2, '--')

    # Axis labels
    plt.xlabel("h")
    plt.ylabel("Error")


#%% Derivative computation

def solve_adjoint(results   : dict,    # structured output from physics.solve_magnetoharmonic
                  df        : callable # function (input "results", output ngsolve symbolic expression)
                  ) -> dict:
    """
    Solve the adjoint problem associated with given finite-element state 
    and objective function.

    This function assembles and solves a linear adjoint system using the
    provided right-hand side `df`, and returns the adjoint field split into
    physically meaningful components (air region and conductor bundles).

    Parameters
    ----------
    results : dict
        Dictionary containing the primal solution and solver information.
        Must include:
        - "info" with the inverse system operator ("Kinv")
        - "bundles" list describing conductor subdomains
        - "test" metadata carried through from the primal solve

    df : callable
        Function that maps `results` to a ngsolve symbolic expression representing 
        the adjoint right-hand side.

    Returns
    -------
    dict
        Dictionary containing the adjoint solution and metadata:
        - "solution": split adjoint fields
            - "a": air / main field component
            - "e": dictionary of bundle-wise components
        - "test": test functions copied from input results
        - "bundles": list of bundle identifiers
        - "info": solver / system information

    Notes
    -----
    - The adjoint system is solved using the precomputed inverse operator
      stored in `results["info"]["Kinv"]`.
    - Complex and real-valued systems are handled separately via Hermitian
      or transpose operators.
    """

    # Assemble adjoint right-hand side
    F = ngs.LinearForm(df(results))

    fes = F.space
    sol = ngs.GridFunction(fes)

    # Assemble system vector
    F = F.Assemble().vec

    # Retrieve precomputed inverse system operator
    Kinv = results["info"]["Kinv"]

    # Solve adjoint system (complex vs real handling)
    if sol.is_complex:
        sol.vec.data = -1 * Kinv.H * F
    else:
        sol.vec.data = -1 * Kinv.T * F

    # ------------------------------------------------------------
    # Package adjoint solution into structured output
    # ------------------------------------------------------------
    adjoint = {"solution": {}}

    # Main (air) component
    adjoint["solution"]["a"] = sol.components[0]

    # Bundle-wise components
    adjoint["solution"]["e"] = {
        bundle: sol.components[i + 1]
        for i, bundle in enumerate(results["bundles"])
    }

    # Propagate metadata
    adjoint["test"] = results["test"]
    adjoint["bundles"] = results["bundles"]
    adjoint["info"] = results["info"]

    return adjoint


def dd_joule_losses(results : dict,          
                    slot    : str   = "slot.*",
                    bonus_intorder : int = 3
                    ) -> ngs.SymbolicBFI :
    """
    Compute the directional derivative of Joule losses.

    This function evaluates the contribution of the current density to the
    Joule loss functional and computes its directional derivative with
    respect to a test perturbation.

    Parameters
    ----------
    results : dict
        Dictionary containing simulation results, including fields required
        to compute current densities and material properties.

    slot : str, optional
        Subdomain selection pattern defining where the Joule losses are
        evaluated (typically stator slot regions).
        
    bonus_intorder : int, optional
        Bonus integration order.

    Returns
    -------
    form
        Weak form representing the directional derivative of Joule losses
        integrated over the specified slot region.

    Notes
    -----
    - The conductivity is regularized with a small offset (1e-300) to avoid
      division by zero.
    - The result is expressed as a finite-element integral form.
    """

    # Current density for primal (solution) state
    j = current_density(results, type="solution")

    # Current density for perturbation (test) state
    j_ = current_density(results, type="test")

    # Electrical conductivity (regularized to avoid division by zero)
    sigma = results["info"]["conductivity"] + 1e-300

    # Returns the directional derivative of losses w.r.t the state
    return ngs.InnerProduct(j, j_) / sigma * ngs.dx(slot, bonus_intorder = bonus_intorder)


def partiald_joule_losses(results :     dict, 
                          adjoint :     dict, 
                          test , # symbolic test function, representing the pertubation direction
                          wrt     :     str ="conductivity", 
                          slot    :     str = "slot.*",
                          bonus_intorder : int = 3,
                          ) -> ngs.LinearForm:
    """
    Compute the partial Fréchet derivative of Joule losses.

    This function evaluates the sensitivity of Joule losses with respect to a
    chosen material parameter (typically conductivity). It assembles a
    finite-element linear form using the primal and adjoint fields.

    Parameters
    ----------
    results : dict
        Primal simulation results, including finite-element spaces and fields.

    adjoint : dict
        Adjoint solution containing field decompositions over domains.

    test : ngs.GridFunction or ngs.LinearForm
        Test function used to build the weak form.

    wrt : str, optional
        Parameter with respect to which the derivative is computed.
        Currently supports:
        - "conductivity" (default)

    slot : str, optional
        Subdomain pattern where the derivative is evaluated.
        
    bonus_intorder : int, optional
        Bonus integration order for conductivity terms.

    Returns
    -------
    ngs.LinearForm
        Assembled finite-element linear form representing the sensitivity.

    Notes
    -----
    - Only conductivity derivatives are implemented.
    - High-order quadrature is used for improved accuracy.
    """

    # Initialize weak form
    df = ngs.LinearForm(test.space)

    if wrt.lower() == "conductivity":

        # Electric field from primal solution
        E = -electric_field(results)

        # Air-domain adjoint component
        pa = adjoint["solution"]["a"]

        # Contribution from conductor bundles
        for bundle in adjoint["bundles"]:
            df += ngs.InnerProduct(
                pa + adjoint["solution"]["e"][bundle],
                E
                ).real * test * ngs.dx(bundle, bonus_intorder = bonus_intorder)

        # Bulk slot contribution
        df += (ngs.InnerProduct(E, E)).real / 2 * test * ngs.dx(slot, bonus_intorder = bonus_intorder)

    # elif wrt.lower() == "resistivity":
    #     TODO

    return df.Assemble()


def d_joule_losses(results  : dict, 
                   val      : ngs.GridFunction, 
                   adjoint  : dict = None, 
                   wrt      : str  ="conductivity",
                   slot    : str = "slot.*",
                   bonus_intorder : int = 3
                   ) -> ngs.LinearForm:
    """
    Compute the total Fréchet derivative of Joule losses.

    This function assembles the full sensitivity of the Joule loss functional
    with respect to a given material parameter (typically conductivity). It
    combines the adjoint contribution and the explicit partial derivative
    into a single finite-element linear form.

    Parameters
    ----------
    results : dict
        Primal simulation results required to evaluate the state and fields.

    val : ngs.GridFunction
        Design variable (used only to access the finite-element space and
        create a test function).

    adjoint : dict, optional
        Precomputed adjoint solution. If None, it is automatically computed.

    wrt : str, optional
        Parameter with respect to which the derivative is computed.
        Currently supports:
        - "conductivity" (default)
        
    bonus_intorder : int, optional
        Bonus integration order for conductivity terms.

    Returns
    -------
    ngs.LinearForm
        Assembled Fréchet derivative of Joule losses.

    Notes
    -----
    - If the adjoint is not provided, it is solved internally.
    - Only conductivity derivatives are currently implemented.
    """

    # ------------------------------------------------------------
    # Conductivity sensitivity (main implemented case)
    # ------------------------------------------------------------
    if wrt.lower() == "conductivity":

        # Solve adjoint system if not provided
        if adjoint is None:
            adjoint = solve_adjoint(results, dd_joule_losses)

        # Test function in design space
        test = val.space.TestFunction()

        # Assemble partial derivative contribution
        df = partiald_joule_losses(results=results,
                                   adjoint=adjoint,
                                   test=test,
                                   wrt="conductivity",
                                   slot=slot,
                                   bonus_intorder = bonus_intorder)

    # elif wrt.lower() == "resistivity":
    #     TODO

    return df

#%% Test

#if __name__ == "__main__" : 
    # TODO
    