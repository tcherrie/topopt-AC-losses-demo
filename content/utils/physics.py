"""Physics utilities

Provide utilities related to the physical formulation and associated solvers.

Functions defined here:
- Curl                      (helper)
- average_property          (helper)
- magnetization_halbach     (helper)
- solve_magnetoharmonic     (main physical solver)
- electric_field            (post-processing)
- current_density           (post-processing)
- joule_losses              (post-processing)
- matrix_arkkio             (post-processing)
- average_torque            (post-processing)


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

import ngsolve as ngs

#%% Helpers

def Curl(u: ngs.GridFunction | ngs.CoefficientFunction
         ) -> ngs.CoefficientFunction:
    """
    Compute the 2D curl operator.

    This function returns the 2D vector curl of a scalar field, corresponding
    to the curl of a z-directed vector quantity into the xy-plane.

    Parameters
    ----------
    u : ngs.GridFunction or ngs.CoefficientFunction
        Scalar field representing the z-component of a vector field.

    Returns
    -------
    ngs.CoefficientFunction
        2D vector field representing Curl(u), defined as:

        b_xy = Curl(a_z)

    Notes
    -----
    - Implements the 2D curl as a 90-degree rotation of the gradient.
    - Equivalent to applying the operator:
    
        [[ 0,  1],
         [-1,  0]] ∇u
    """
    return ngs.CF(((0, 1), (-1, 0)), dims=(2, 2)) * ngs.grad(u)


def average_property(property: ngs.GridFunction | ngs.CoefficientFunction,
                     results: dict,
                     zone: str = ".*"
                     ) -> float:
    """
    Compute the spatial average of a field over a given mesh region.

    This function evaluates the mean value of a scalar (or scalar-valued)
    finite-element field over a selected geometric zone.

    Parameters
    ----------
    property : ngs.GridFunction or ngs.CoefficientFunction
        Field to be averaged over the domain.

    results : dict
        Simulation results dictionary containing at least the FESpace
        and mesh information under `results["info"]["fes"]`.

    zone : str, optional
        Material or region selector (regex-style). Default is ".*" (whole domain).

    Returns
    -------
    float
        Spatial average of the given property over the selected zone.

    Notes
    -----
    - The average is computed as:
        ⟨f⟩ = (∫_Ω f dx) / (∫_Ω 1 dx)
    - The integration is performed using NGSolve integration utilities.
    """

    mesh = results["info"]["fes"].mesh

    # Compute total area/volume of the selected region
    surface_zone = ngs.Integrate(1, mesh.Materials(zone))

    # Compute integral of the field over the region and normalize
    return ngs.Integrate(property, mesh.Materials(zone)) / surface_zone



def magnetization_halbach(br: float = 1,
                          mu: float = 4e-7 * ngs.pi,
                          p: int = 4
                          ) -> ngs.CoefficientFunction:
    """
    Construct a complex-valued Halbach magnetization field.

    This function defines the magnetization of a Halbach array in polar
    coordinates, producing a rotating magnetization vector with harmonic
    order controlled by the pole pair number.

    Parameters
    ----------
    br : float, optional
        Remanent flux density magnitude.

    mu : float, optional
        Magnetic permeability (default is vacuum permeability).

    p : int, optional
        Number of pole pairs defining the spatial harmonic order.

    Returns
    -------
    ngs.CoefficientFunction
        Complex-valued 2D magnetization field in Cartesian components.

    Notes
    -----
    - The field is expressed in polar coordinates using the angle
      alpha = atan2(y, x).
    - The Halbach distribution is encoded using complex exponentials,
      representing a rotating magnetization pattern.
    - Commonly used to model ideal permanent magnet arrays in rotating
      electrical machines.
    """

    alpha = ngs.atan2(ngs.y, ngs.x)

    return br / mu * ngs.CF((ngs.exp(-1j * (p - 1) * alpha),
                             ngs.exp(-1j * ((p - 1) * alpha + ngs.pi / 2)) )
    )

#%% Main physical solver

def solve_magnetoharmonic(
    fes: ngs.FESpace,  # finite element space
    reluctivity: ngs.GridFunction | ngs.CoefficientFunction,  # magnetic reluctivity
    magnetization: ngs.GridFunction | ngs.CoefficientFunction,  # complex magnetization
    frequency: float,   # electrical frequency
    supply: dict,       # supply of electrical conductors
    conductivity: ngs.GridFunction | ngs.CoefficientFunction | float = 6e7,  # conductivity
    Kinv=None,  # optional precomputed inverse system matrix
    solver: str = "pardiso",  # linear solver type
    bonus_intorder : int = 3,       # bonus intorder for conductivity and magnetizationterms
    verbose: int = 0,  # for controlling print statements
    taskmanager: bool = True, # for paralelizing assembly process
    ) -> dict:
    """
    Solve a magneto-quasistatics problem in the frequency domain.

    This function assembles and solves a coupled finite-element system for
    the magnetic vector potential and bundle-wise electric potentials in
    conducting regions, including eddy-current effects in lectrical conductors
    and imposed supply currents.

    Parameters
    ----------
    fes : ngs.FESpace
        Finite element space for the coupled magneto-harmonic system.

    reluctivity : ngs.GridFunction or ngs.CoefficientFunction
        Magnetic reluctivity (1/μ), possibly spatially varying.
        Magnetic materials are assumed linear.

    magnetization : ngs.GridFunction or ngs.CoefficientFunction
        Complex-valued magnetization source term (e.g., permanent magnets).

    frequency : float
        Electrical excitation frequency [Hz].

    supply : dict
        Dictionary mapping conductor bundle names to imposed currents,
        so that different types of windings can be simulated.

    conductivity : ngs.GridFunction or ngs.CoefficientFunction, optional
        Electrical conductivity distribution.

    Kinv : optional
        Precomputed inverse of the system matrix to accelerate solves.

    solver : str, optional
        Direct solver backend used for matrix inversion (default: "pardiso").

    bonus_intorder : int, optional
        Bonus integration order for conductivity terms.

    verbose : int, optional
        Verbosity level for assembly and solve progress.

    taskmanager : bool, optional
        If True, enables NGSolve TaskManager for parallel assembly.

    Returns
    -------
    results : dict
        Dictionary containing:
        - "solution": solved fields
            - "a": magnetic vector potential
            - "e": bundle electric potentials
        - "test": test functions used in weak formulation
        - "bundles": list of conductor bundle identifiers
        - "info": metadata (FESpace, material parameters, solver info)

    Notes
    -----
    - The formulation imposes total current via Lagrange multiplier
    - If Kinv is provided, the factorization step is skipped which 
      accelerates the computation a lot (e.g, for different current excitations)
    """

    jw = 1j * 2 * ngs.pi * frequency

    if verbose >= 1:
        print("Setup function space... ", end="")

    # Identify conductor bundles
    bundles = supply.keys()

    mesh = fes.mesh

    # Normalize supplied currents by bundle volume
    Jcplx = {
        bundle: supply[bundle] / ngs.Integrate(1, mesh.Materials(bundle))
        for bundle in bundles
    }

    # Extend FE space with Lagrange multipliers (bundle constraints)
    for _ in bundles:
        fes *= ngs.NumberSpace(mesh, complex=True)

    # Define trial and test functions
    trials = fes.TrialFunction()
    tests = fes.TestFunction()

    a, a_ = trials[0], tests[0]

    e = {bundle: trials[i + 1] for i, bundle in enumerate(bundles)}
    e_ = {bundle: tests[i + 1] for i, bundle in enumerate(bundles)}

    if verbose >= 1:
        print("Done!")

    # ------------------------------------------------------------
    # Assemble system matrix
    # ------------------------------------------------------------
    if Kinv is None:

        if verbose >= 1:
            print("Assemble matrix... ", end="")

        bf = Curl(a_) * reluctivity * Curl(a) * ngs.dx

        # Eddy-current + constraint coupling in each bundle
        for bundle in bundles:
            bf += a_ * conductivity * (jw * a + e[bundle]) * ngs.dx(bundle, bonus_intorder = bonus_intorder)
            bf += e_[bundle] * conductivity * (jw * a + e[bundle]) * ngs.dx(bundle, bonus_intorder = bonus_intorder)

        # Assemble matrix
        if taskmanager:
            with ngs.TaskManager():
                K = ngs.BilinearForm(bf).Assemble().mat
        else:
            K = ngs.BilinearForm(bf).Assemble().mat

        if verbose >= 1:
            print("Done!")

        # Factorize system
        if verbose >= 1:
            print(f"Matrix decomposition with {solver}... ", end="")

        Kinv = K.Inverse(fes.FreeDofs(), inverse=solver)

        if verbose >= 1:
            print("Done!")

    # ------------------------------------------------------------
    # Assemble right-hand side
    # ------------------------------------------------------------
    if verbose >= 1:
        print("Assemble right hand side... ", end="")

    lf = Curl(a_) * magnetization * ngs.dx(bonus_intorder = bonus_intorder)

    for bundle in bundles:
        lf += -e_[bundle] * Jcplx[bundle] * ngs.dx(bundle)

    F = ngs.LinearForm(lf).Assemble().vec

    if verbose >= 1:
        print("Done!")

    # ------------------------------------------------------------
    # Solve linear system
    # ------------------------------------------------------------
    if verbose >= 1:
        print("Solve the problem... ", end="")

    sol = ngs.GridFunction(fes)
    sol.vec.data += Kinv * F

    if verbose >= 1:
        print("Done!")

    # ------------------------------------------------------------
    # Package results
    # ------------------------------------------------------------
    if verbose >= 1:
        print("Pack the results... ", end="")

    results = {
        "solution": {
            "a": sol.components[0],
            "e": {
                bundle: sol.components[i + 1]
                for i, bundle in enumerate(bundles)
            },
        },
        "test": {"a": a_, "e": e_},
        "bundles": bundles,
        "info": {
            "fes": fes,
            "reluctivity": reluctivity,
            "magnetization": magnetization,
            "frequency": frequency,
            "supply": supply,
            "conductivity": conductivity,
            "Kinv": Kinv,
            "solver": solver,
        },
    }

    if verbose >= 1:
        print("Done!")

    return results

#%% Post-processing

def electric_field(results: dict,              # result of solve_magnetoharmonic
                   type: str = "solution"      # "solution" or "test" for directional derivative
                   ) -> ngs.CoefficientFunction:
    """
    Compute the electric field in conducting regions by post-processing.

    This function reconstructs the complex electric field from the magnetic
    vector potential and bundle-wise electric potentials obtained from a
    magneto-harmonic finite element solve.

    Parameters
    ----------
    results : dict
        Output dictionary from `solve_magnetoharmonic`, containing the
        solution fields and metadata.

    type : str, optional
        Specifies which fields to use:
        - "solution": use primal solution fields
        - "test": use adjoint/test fields for sensitivity analysis

    Returns
    -------
    ngs.CoefficientFunction
        Complex electric field distribution, restricted to conducting regions.

    Notes
    -----
    - The electric field is computed as:
        E = -jω A - e_bundle
    - A conductivity mask is applied to restrict the field to conductors
      and avoid division by zero.
    """

    jw = 1j * 2 * ngs.pi * results["info"]["frequency"]

    # Time-harmonic contribution from magnetic vector potential
    E = -jw * results[type]["a"]

    mesh = results["info"]["fes"].mesh

    # Add bundle electric potential contributions
    for bundle in results[type]["e"].keys():
        e = results[type]["e"][bundle]
        E += -e * mesh.MaterialCF({bundle: 1})

    # Apply conductivity mask (avoid division by zero)
    sigma = results["info"]["conductivity"]

    return E * sigma / (sigma + 1e-300)

def current_density(results: dict,          # result of solve_magnetoharmonic
                    type: str = "solution"  # "solution" or "test" for directional derivative
                    ) -> ngs.CoefficientFunction:
    """
    Compute the electric current density by post-processing.

    This function evaluates the conductive current density from the electric
    field obtained in a magneto-harmonic finite element simulation.

    Parameters
    ----------
    results : dict
        Output dictionary from `solve_magnetoharmonic`, containing solution
        fields and material properties.

    type : str, optional
        Specifies which field set to use:
        - "solution": use primal solution fields
        - "test": use test fields for sensitivity analysis

    Returns
    -------
    ngs.CoefficientFunction
        Complex current density field J = σ E (local Ohm's law)
    """

    sigma = results["info"]["conductivity"]

    return sigma * electric_field(results, type)

def joule_losses(results : dict) -> float:
    """
    Compute the total Joule (ohmic) losses by post-processing.

    This function evaluates the resistive power dissipation in conducting
    regions based on the current density obtained from a magneto-harmonic
    finite element simulation.

    Parameters
    ----------
    results : dict
        Output dictionary from `solve_magnetoharmonic`, containing solution
        fields and material properties.

    Returns
    -------
    float
        Total Joule losses (time-averaged power dissipation).

    Notes
    -----
    - The Joule losses are computed as:
        P = 1/2 ∫_Ω (|J|² / σ) dx
      where J is the complex current density.
    - The factor 1/2 accounts for time-averaging in harmonic regime; 
    we assume phasors modules are amplitude and not RMS values.
    - A small regularization term is added to σ to avoid division by zero.
    """

    # Current density from post-processing
    j = current_density(results)

    # Conductivity with numerical safety offset
    sigma = results["info"]["conductivity"] + 1e-300

    mesh = results["info"]["fes"].mesh

    # Quadrature order adapted to solution polynomial order
    order = 2 * results["solution"]["a"].space.globalorder + 1

    # Time-averaged Joule losses: 1/2 ∫ |J|^2 / σ dx
    P = ngs.Integrate(
        ngs.InnerProduct(j, j).real / sigma,
        mesh,
        order=order) / 2

    # Imaginary part is zero in theory; only real part is used explicitly
    return P

def matrix_arkkio() -> ngs.CoefficientFunction :
    """
    Construct the Arkkio torque weighting matrix.

    This operator defines the 2D geometric weighting matrix used in
    Arkkio's method for torque computation in electrical machines.
    It depends only on spatial coordinates (x, y).

    Returns
    -------
    ngs.CoefficientFunction
        2×2 matrix field used to weight the magnetic flux density
        in the torque integral.

    Notes
    -----
    - Implements the standard Arkkio geometric tensor:
        Q = (1/r) * [ -xy, -(y² - x²)/2 ; -(y² - x²)/2, xy ]
    - r = sqrt(x² + y²)
    - Used for torque evaluation in the airgap region.
    """

    xy, x2, y2 = ngs.x * ngs.y, ngs.x**2, ngs.y**2
    r = ngs.sqrt(x2 + y2)

    Q11 = -xy / r
    Q21 = -(y2 - x2) / (2 * r)

    return ngs.CF(((Q11, Q21), (Q21, -Q11)), dims=(2, 2))


def average_torque(results : dict, 
                   airgap : str = "airgap_rotor",
                   L : float = 1. # axial length
                   ) -> float:
    """
    Compute the average electromagnetic torque using Arkkio's method.

    This function evaluates the electromagnetic torque in the airgap
    region based on the magnetic flux density and a geometric weighting
    tensor.

    Parameters
    ----------
    results : dict
        Output dictionary from `solve_magnetoharmonic`, containing the
        magnetic vector potential solution.

    airgap : str, optional
        Material name corresponding to the airgap region where torque
        is evaluated.
    
    L :float, optional
        Axial length of the machine

    Returns
    -------
    float
        Average electromagnetic torque.

    Notes
    -----
    - Uses Arkkio's method:
        T = (L π / (μ₀ S)) ∫ (B · (Q B)) dΩ
        see this paper: https://arxiv.org/pdf/2511.07217
    - S is the airgap surface area.
    """

    # Quadrature order adapted to FE polynomial degree
    order = 2 * results["solution"]["a"].space.globalorder + 1

    mesh = results["info"]["fes"].mesh

    # Airgap normalization area
    S = ngs.Integrate(1, mesh.Materials(airgap))

    # Magnetic flux density
    b = Curl(results["solution"]["a"])

    mu0 = 4e-7 * ngs.pi

    # Arkkio torque formula
    integrand = ngs.InnerProduct(b, (matrix_arkkio() * b))
    factor = L * ngs.pi / (S * mu0) 
    return factor * ngs.Integrate(integrand , mesh.Materials(airgap), order=order).real