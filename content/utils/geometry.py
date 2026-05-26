"""Geometry utilities

Provide utilities to generate the machine geometry and mesh.

Functions defined here:
- plot_points               (debug)
- plot_lines                (debug)
- find_tangent_intersection (helper)
- machine_mesh              (main function)

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
import matplotlib.pyplot as plt
import numpy as np

from netgen.geom2d import SplineGeometry


#%% Debug

def plot_points(points : tuple[float, float]
                )-> None:
    """
    Plot a set of points on a 2D graph with labels (used for debugging).

    This function visualizes a list of 2D points using matplotlib, displaying each point
    as a red cross marker and labeling them sequentially (p0, p1, etc.). The plot axes
    are set to equal scale for accurate geometric representation.

    Parameters:
    -----------
    points : list of tuples or list of lists
        A collection of 2D points where each point is either:
        - A tuple (x, y)
        - A list, so that the first element of the list is a tuple with coordinates of the points.

    Returns:
    --------
    None
        This function displays a plot and does not return any value.
    """
    for i, xy in enumerate(points):
        if type(xy) is not tuple:
            xy = xy[0]  # Convert list to tuple if needed

        plt.scatter(xy[0], xy[1], c="r", marker="+")
        plt.text(xy[0], xy[1], "p" + str(i))

    plt.axis("equal")  # Ensure equal scaling for both axes


def plot_lines(pnts: list,
               curves: list,
               fac: float = 5e-3
               ) -> None:
    """
    Plot geometry curves and display adjacent domain numbers.

    This debugging utility visualizes line and spline curves used in the
    geometry construction. The domain identifiers located on each side of
    the curve are displayed in red.

    Parameters
    ----------
    pnts : list
        List of geometry points.

    curves : list
        List of curves with associated domain information.

    fac : float, optional
        Offset factor used to position domain labels away from the curves.

    Returns
    -------
    None
        Displays the plot using matplotlib.
    """

    for i in range(len(curves)):

        # Extract curve definition
        curve = curves[i][0]

        # ------------------------------------------------------------
        # Straight line segment
        # ------------------------------------------------------------
        if curve[0] == "line":

            # Extract endpoints
            if type(pnts[curve[1]]) is tuple:
                p1 = pnts[curve[1]]
                p2 = pnts[curve[2]]
            else:
                p1 = pnts[curve[1]][0]
                p2 = pnts[curve[2]][0]

            # Draw line
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

            # Compute unit normal vector for label positioning
            d = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            d = d / np.linalg.norm(d)
            dLeft = np.array([[0, -1], [1, 0]]) @ d

            # Midpoint of the segment
            pavg = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

            # Display left/right domain numbers
            plt.text(
                pavg[0] + fac * dLeft[0],
                pavg[1] + fac * dLeft[1],
                str(curves[i][1]["leftdomain"]),
                color="r"
            )

            plt.text(
                pavg[0] - fac * dLeft[0],
                pavg[1] - fac * dLeft[1],
                str(curves[i][1]["rightdomain"]),
                color="r"
            )

        # ------------------------------------------------------------
        # Spline segment (approximated buy 2 straight lines)
        # ------------------------------------------------------------
        elif curve[0] == "spline3":

            # Extract spline control points
            if type(pnts[curve[1]]) is tuple:
                p1 = pnts[curve[1]]
                p2 = pnts[curve[2]]
                p3 = pnts[curve[3]]
            else:
                p1 = pnts[curve[1]][0]
                p2 = pnts[curve[2]][0]
                p3 = pnts[curve[3]][0]

            # Draw spline polyline
            plt.plot(
                [p1[0], p2[0], p3[0]],
                [p1[1], p2[1], p3[1]]
            )

            # Approximate tangent direction
            d = np.array([p3[0] - p1[0], p3[1] - p1[1]])
            d = d / np.linalg.norm(d)

            # Left normal vector
            dLeft = np.array([[0, -1], [1, 0]]) @ d

            # Display left/right domain numbers near spline midpoint
            plt.text(
                p2[0] + fac * dLeft[0],
                p2[1] + fac * dLeft[1],
                str(curves[i][1]["leftdomain"]),
                color="r"
            )

            plt.text(
                p2[0] - fac * dLeft[0],
                p2[1] - fac * dLeft[1],
                str(curves[i][1]["rightdomain"]),
                color="r"
            )

    # Preserve geometric proportions
    plt.axis("equal")


#%% Helper

def find_tangent_intersection(p1:       tuple[float, float],
                              p2:       tuple[float, float],
                              center:   tuple[float, float] = (0., 0.)):
    """
    Calculate the intersection point of two tangent lines to a circle.

    This function computes the point where the tangent lines from two points (p1 and p2)
    on a circle intersect. The circle is defined by its center point and the fact that
    both input points lie on its circumference.

    Parameters:
    -----------
    p1 : tuple
        Coordinates of the first point on the circle (x1, y1).
    p2 : tuple
        Coordinates of the second point on the circle (x2, y2).
    center : tuple, optional
        Coordinates of the circle's center (cx, cy). Default is (0, 0).

    Returns:
    --------
    tuple or None
        The intersection point of the two tangent lines as a (x, y) tuple.
        Returns None if the tangent lines are parallel (degenerate case).

    Notes:
    ------
    - The function assumes both input points lie on the circumference of the circle.
    - The solution involves finding the perpendicular vectors to the radii and solving
      for their intersection point.
    """
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = center

    # Radius vectors from center to points
    r1x, r1y = x1 - cx, y1 - cy
    r2x, r2y = x2 - cx, y2 - cy

    # Tangent directions (perpendicular to radii)
    # Rotation of vector (a, b) by 90 degrees gives (-b, a)
    d1x, d1y = -r1y, r1x
    d2x, d2y = -r2y, r2x

    # Solve intersection: p1 + t*d1 = p2 + u*d2
    det = d1x * d2y - d1y * d2x

    if abs(det) < 1e-12:
        # Parallel tangents (degenerate case)
        return None

    dx = x2 - x1
    dy = y2 - y1

    t = (dx * d2y - dy * d2x) / det

    ix = x1 + t * d1x
    iy = y1 + t * d1y

    return (ix, iy)


#%% mesh

def machine_mesh(# Primitives
                p : int = 4,                # number of pole pairs >= 2
                Rint : float = 100*1e-3,    # shaft radius [m]
                Rext : float = 150*1e-3,    # external radius of the yoke [m]
                e : float = 0.5e-3,         # airgap thickness [m]
                # Relative geometric parameters in (0, 1)
                kRe : float = 0.3,          # relative airgap radius
                kA : float = 0.5,           # stator flux path thickness
                kShoe : float = 0.05,       # relative stator shoe thickness
                kRint_air : float = 0.8,    # relative internal air radius
                kRext_air : float = 1.1,    # relative external air radius (>1)
                # Bundles parameters
                bundles_per_half_slot : int = 1, # number of bundles per half slot
                kBundleR : float = 0.9,     # relative radial extension of conductor
                kBundleTh : float = 0.9,    # relative orthoradial extension of conductor
                # Elements sizes
                hAirOut : float = 10e-3,    # outside the machine [m]
                hRotor : float = 3e-3,      # within the rotor [m]
                hSlot :  float = 2e-3,      # within the stator slots [m]
                hBundle : float = 2e-3,     # within the conductor bundles [m]
                hShoe : float = 2e-3,      # within the slots shoes [m]
                hAirgap : float = 0.5e-3,   # within the air gap [m]
                hStator : float = 3e-3,     # within the stator core [m]
                hCorner : float = 0.5e-3,   # at the interior stator corners (singularity) [m]
                # Domain names
                airInner_label : str = "air_inner",
                airOuter_label : str = "air_outer",
                rotor_label : str = "rotor",
                airgapRotor_label : str = "airgap_rotor",
                airgapStator_label : str = "airgap_stator",
                coreStator_label : str = "core_stator",
                slot11_label : str = "slot11",
                slot12_label : str = "slot12",
                slot21_label : str = "slot21",
                slot22_label : str = "slot22",
                slot31_label : str = "slot31",
                slot32_label : str = "slot32",
                shoe11_label : str = "shoe11",
                shoe12_label : str = "shoe12",
                shoe21_label : str = "shoe21",
                shoe22_label : str = "shoe22",
                shoe31_label : str = "shoe31",
                shoe32_label : str = "shoe32",
                bundle_label : str = "bundle",
                # Boundary names
                shaft_bnd_label :  str = "shaft",
                out_bnd_label :  str = "out",
                slidingBand_bnd_label :  str = "slinding_band",
                master1_bnd_label :  str = "master1",
                master2_bnd_label :  str = "master2",
                master3_bnd_label :  str = "master3",
                master4_bnd_label :  str = "master4",
                master5_bnd_label :  str = "master5",
                master6_bnd_label :  str = "master6",
                master7_bnd_label :  str = "master7",
                minion1_bnd_label :  str = "minion1",
                minion2_bnd_label :  str = "minion2",
                minion3_bnd_label :  str = "minion3",
                minion4_bnd_label :  str = "minion4",
                minion5_bnd_label :  str = "minion5",
                minion6_bnd_label :  str = "minion6",
                minion7_bnd_label :  str = "minion7",
                # Debug
                debug : bool = False
                ):
    """
    Generate the finite element mesh of a single pole sector of a rotating
    electrical machine.

    This function constructs the full 2D parametric geometry of a machine pole,
    including the rotor, stator, air gap, stator teeth, slot shoes, winding
    slots, optional conductor bundles, and surrounding air regions. The geometry
    is discretized using spline-based boundaries and returned as an NGSolve mesh.

    The mesh is intended for electromagnetic finite element simulations with
    periodic boundary conditions and sliding-band interfaces between rotor and
    stator domains.

    Parameters
    ----------
    p : int, optional
        Number of pole pairs in the machine, should be greater than or equal to 2.

    Rint : float, optional
        Radius of the shaft [m].

    Rext : float, optional
        External radius of the stator yoke [m].

    e : float, optional
        Air-gap thickness [m].

    kRe : float, optional
        Relative position of the air-gap center radius within the machine.

    kA : float, optional
        Relative stator tooth angular width factor.

    kShoe : float, optional
        Relative thickness of the slot shoe region.

    kRint_air : float, optional
        Relative inner air radius with respect to ``Rint``.

    kRext_air : float, optional
        Relative outer air radius with respect to ``Rext``.

    bundles_per_half_slot : int, optional
        Number of conductor bundles inserted in each half-slot.
        If equal to 1, the slot itself represents the conductor region.

    kBundleR : float, optional
        Radial filling factor of conductor bundles inside the slots.

    kBundleTh : float, optional
        Tangential filling factor of conductor bundles inside the slots.

    hAirOut : float, optional
        Target mesh size in the outer air region [m].

    hRotor : float, optional
        Target mesh size in the rotor region [m].

    hSlot : float, optional
        Target mesh size in the stator slots [m].

    hBundle : float, optional
        Target mesh size inside conductor bundles [m].

    hShoe : float, optional
        Target mesh size in the slot shoe regions [m].

    hAirgap : float, optional
        Target mesh size in the air-gap regions [m].

    hStator : float, optional
        Target mesh size in the stator core [m].

    hCorner : float, optional
        Local mesh refinement size near stator corner singularities [m].

    airInner_label, airOuter_label, rotor_label, airgapRotor_label, \
    airgapStator_label, coreStator_label : str, optional
        Material labels assigned to the corresponding machine regions.

    slotXX_label : str, optional
        Labels assigned to the stator slot domains.

    shoeXX_label : str, optional
        Labels assigned to the stator shoe domains.

    bundle_label : str, optional
        Base label used for conductor bundle domains.

    shaft_bnd_label : str, optional
        Boundary condition label for the shaft boundary.

    out_bnd_label : str, optional
        Boundary condition label for the outer boundary.

    slidingBand_bnd_label : str, optional
        Boundary label associated with the rotor/stator sliding interface.

    masterX_bnd_label, minionX_bnd_label : str, optional
        Labels used for periodic boundary condition pairing.

    debug : bool, optional
        If ``True``, display intermediate geometry and point visualizations.

    Returns
    -------
    ngsolve.Mesh
        Generated finite element mesh of the machine pole sector.

    Notes
    -----
    - The geometry is constructed using spline segments and circular arcs.
    - Periodic boundaries are automatically defined for sector symmetry.
    - Optional conductor bundles are generated as additional subdomains inside
      the winding slots.
    - Mesh refinement is locally enforced in the air gap and near geometric
      singularities to improve electromagnetic field accuracy.

    Dependencies
    ------------
    This function relies on:
    - NumPy
    - NGSolve / Netgen
    - ``SplineGeometry``
    - ``find_tangent_intersection()``
    - Optional plotting utilities for debug visualization.

    Examples
    --------
    Generate a default machine mesh:

    >>> mesh = mesh_machine()

    Generate a refined mesh with multiple conductor bundles:

    >>> mesh = mesh_machine(
    ...     bundles_per_half_slot=3,
    ...     hAirgap=2e-4,
    ...     debug=True
    ... )
    """

    if bundles_per_half_slot == 1:
        slot11_label = slot11_label + "_" + bundle_label + "0"
        slot12_label = slot12_label + "_" + bundle_label + "0"
        slot21_label = slot21_label + "_" + bundle_label + "0"
        slot22_label = slot22_label + "_" + bundle_label + "0"
        slot31_label = slot31_label + "_" + bundle_label + "0"
        slot32_label = slot32_label + "_" + bundle_label + "0"

    
    ##################################################################################
    # Domains numbers
    domainAirInner = 1
    domainRotor = 2 
    domainAirgapRotor = 3
    domainAirgapStator = 4
    domainCoreStator= 5
    domainSlot11 = 6
    domainSlot12 = 7
    domainSlot21 = 8
    domainSlot22 = 9
    domainSlot31 = 10
    domainSlot32 = 11
    domainShoe11 = 12
    domainShoe12 = 13
    domainShoe21 = 14
    domainShoe22 = 15
    domainShoe31 = 16
    domainShoe32 = 17
    domainAirOut= 18

    ##################################################################################
    # Additional geometric quantities
    Rint_air = kRint_air * Rint
    thPole = np.pi / p
    Re = Rint + e/2 + kRe * (Rext-Rint-e)
    Rext_rotor = Re-e/2
    Rint_stator = Re+e/2
    thTooth = kA*thPole/3
    a = 2*np.sin(thTooth/2)*Rint_stator
    thHalfSlot = thPole/6
    Ryoke = Rext-a
    Rext_air = kRext_air*Rext
    Rshoe = Rint_stator + kShoe * (Ryoke - Rint_stator)

    ##################################################################################
    # Points

    pnts = [[(Rint_air,0), {}],
            [(Rint, 0), {}],
            [(Rint*np.cos(thPole), Rint*np.sin(thPole)), {}],
            [(Rint_air*np.cos(thPole), Rint_air*np.sin(thPole)), {}]
    ]
    pnts.append([find_tangent_intersection(pnts[0][0],pnts[3][0]), {}])
    pnts.append([find_tangent_intersection(pnts[1][0],pnts[2][0]), {}])

    pntsrotor = [ [(Rext_rotor,0), {}],
            [(Rext_rotor*np.cos(thPole), Rext_rotor*np.sin(thPole)), {}]
    ]
    pntsrotor.append([find_tangent_intersection(pntsrotor[0][0],pntsrotor[1][0]), {}])
    pnts += pntsrotor

    pntsairgap1 = [ [(Re,0), {}],
            [(Re*np.cos(thPole), Re*np.sin(thPole)), {}]
    ]
    pntsairgap1.append([find_tangent_intersection(pntsairgap1[0][0],pntsairgap1[1][0]), {}])
    pnts +=  pntsairgap1

    pnts2 =[[(Rint_stator,0), {}],
            [(Rint_stator*np.cos(thTooth/2), a/2 ), {}],
            [(Rint_stator*np.cos(thHalfSlot), Rint_stator*np.sin(thHalfSlot) ), {}],
            [(Rint_stator*np.cos(thPole/3 - thTooth/2), Rint_stator*np.sin(thPole/3 - thTooth/2) ), {}],
            [(Rint_stator*np.cos(thPole/3 + thTooth/2), Rint_stator*np.sin(thPole/3 + thTooth/2) ), {}],
            [(Rint_stator*np.cos(thHalfSlot+thPole/3), Rint_stator*np.sin(thHalfSlot+thPole/3) ), {}],
            [(Rint_stator*np.cos(2*thPole/3 - thTooth/2), Rint_stator*np.sin(2*thPole/3 - thTooth/2) ), {}],
            [(Rint_stator*np.cos(2*thPole/3 + thTooth/2), Rint_stator*np.sin(2*thPole/3 + thTooth/2) ), {}],
            [(Rint_stator*np.cos(thPole - thHalfSlot), Rint_stator*np.sin(thPole - thHalfSlot) ), {}],
            [(Rint_stator*np.cos(thPole - thTooth/2), Rint_stator*np.sin(thPole - thTooth/2) ), {}],
            [(Rint_stator*np.cos(thPole), Rint_stator*np.sin(thPole) ), {}]
    ]

    pnts +=pnts2

    pntSpline = [[find_tangent_intersection(pnts2[i][0],pnts2[i+1][0]), {}] for i in range(len(pnts2)-1)]

    pnts += pntSpline

    dThetaY = np.asin(a/2/Ryoke)
    pnts3 =[[(Ryoke,0), {}],
            [(Ryoke*np.cos(dThetaY), a/2), {"maxh": hCorner}],
            [(Ryoke*np.cos(thHalfSlot), Ryoke*np.sin(thHalfSlot)), {}],
            [(Ryoke*np.cos(thPole/3-dThetaY), Ryoke*np.sin(thPole/3-dThetaY)), {"maxh": hCorner}],
            [(Ryoke*np.cos(thPole/3+dThetaY), Ryoke*np.sin(thPole/3+dThetaY)), {"maxh": hCorner}],
            [(Ryoke*np.cos(thPole/3+thHalfSlot), Ryoke*np.sin(thPole/3+thHalfSlot)), {}],
            [(Ryoke*np.cos(2*thPole/3-dThetaY), Ryoke*np.sin(2*thPole/3-dThetaY)), {"maxh": hCorner}],
            [(Ryoke*np.cos(2*thPole/3+dThetaY), Ryoke*np.sin(2*thPole/3+dThetaY)), {"maxh": hCorner}],
            [(Ryoke*np.cos(2*thPole/3+thHalfSlot), Ryoke*np.sin(2*thPole/3+thHalfSlot)), {}],
            [(Ryoke*np.cos(thPole-dThetaY), Ryoke*np.sin(thPole-dThetaY)), {"maxh": hCorner}],
            [(Ryoke*np.cos(thPole),Ryoke*np.sin(thPole)), {}],
    ]

    pnts += pnts3

    splinePnts3 = [[find_tangent_intersection(pnts3[1][0],pnts3[2][0]), {}],
                [find_tangent_intersection(pnts3[2][0],pnts3[3][0]), {}],
                [find_tangent_intersection(pnts3[4][0],pnts3[5][0]), {}],
                [find_tangent_intersection(pnts3[5][0],pnts3[6][0]), {}],
                [find_tangent_intersection(pnts3[7][0],pnts3[8][0]), {}],
                [find_tangent_intersection(pnts3[8][0],pnts3[9][0]), {}],]

    pnts += splinePnts3

    pnts4 =[[(Rext,0), {}],
            [(Rext*np.cos(thPole), Rext*np.sin(thPole)), {}],
            [(Rext_air,0), {}],
            [(Rext_air*np.cos(thPole), Rext_air*np.sin(thPole)), {}],
    ]
    pnts4.append([find_tangent_intersection(pnts4[0][0],pnts4[1][0]), {}])
    pnts4.append([find_tangent_intersection(pnts4[2][0],pnts4[3][0]), {}])

    pnts += pnts4
    dThetaShoes = np.asin(a/2/Rshoe)
    pntsShoes = [[(Rshoe*np.cos(dThetaShoes), a/2), {}],
                [(Rshoe*np.cos(thHalfSlot), Rshoe*np.sin(thHalfSlot)), {}],
                [(Rshoe*np.cos(thPole/3 - dThetaShoes), Rshoe * np.sin(thPole/3 - dThetaShoes)), {}],
                [(Rshoe*np.cos(thPole/3 + dThetaShoes), Rshoe * np.sin(thPole/3 + dThetaShoes)), {}],
                [(Rshoe*np.cos(thHalfSlot+thPole/3), Rshoe*np.sin(thHalfSlot+thPole/3)), {}],
                [(Rshoe*np.cos(2*thPole/3 - dThetaShoes), Rshoe * np.sin(2*thPole/3 - dThetaShoes)), {}],
                [(Rshoe*np.cos(2*thPole/3 + dThetaShoes), Rshoe * np.sin(2*thPole/3 + dThetaShoes)), {}],
                [(Rshoe*np.cos(thHalfSlot+2*thPole/3), Rshoe*np.sin(thHalfSlot+2*thPole/3)), {}],
                [(Rshoe*np.cos(thPole - dThetaShoes), Rshoe * np.sin(thPole - dThetaShoes)), {}],
    ]

    pnts += pntsShoes

    pntsShoesSpline = [[find_tangent_intersection(pntsShoes[0][0],pntsShoes[1][0]), {}],
                    [find_tangent_intersection(pntsShoes[1][0],pntsShoes[2][0]), {}],
                    [find_tangent_intersection(pntsShoes[3][0],pntsShoes[4][0]), {}],
                    [find_tangent_intersection(pntsShoes[4][0],pntsShoes[5][0]), {}],
                    [find_tangent_intersection(pntsShoes[6][0],pntsShoes[7][0]), {}],
                    [find_tangent_intersection(pntsShoes[7][0],pntsShoes[8][0]), {}],
    ]
    pnts += pntsShoesSpline

    ##################################################################################
    # Curves

    periodic = [[["line",0,1], {"leftdomain": domainAirInner, "rightdomain": 0, "maxh": hAirOut, "bc": master1_bnd_label}],
               [["line",1,6], { "leftdomain": domainRotor, "rightdomain": 0, "maxh": hRotor, "bc": master2_bnd_label}],
               [["line",6,9], { "leftdomain": domainAirgapRotor, "rightdomain": 0, "maxh": hAirgap, "bc": master3_bnd_label}],
               [["line",9,12], { "leftdomain": domainAirgapStator, "rightdomain": 0, "maxh": hAirgap, "bc": master4_bnd_label}],
               [["line",12,33], { "leftdomain": domainCoreStator, "rightdomain": 0, "maxh": hStator, "bc": master5_bnd_label}],
               [["line",33,50], { "leftdomain": domainCoreStator, "rightdomain": 0, "maxh": hStator, "bc": master6_bnd_label}],
               [["line",50,52], {"leftdomain": domainAirOut, "rightdomain": 0, "maxh": hAirOut, "bc": master7_bnd_label}],

               [["line",3,2], {"rightdomain": domainAirInner, "leftdomain": 0, "maxh": hAirOut, "copy" : 0, "bc": minion1_bnd_label}],
               [["line",2,7], { "rightdomain": domainRotor, "leftdomain": 0, "maxh": hRotor, "copy" : 1, "bc": minion2_bnd_label}],
               [["line",7,10], { "rightdomain": domainAirgapRotor, "leftdomain": 0, "maxh": hAirgap, "copy" : 2, "bc": minion3_bnd_label}],
               [["line",10,22], { "rightdomain": domainAirgapStator, "leftdomain": 0, "maxh": hAirgap, "copy" : 3, "bc": minion4_bnd_label}],
               [["line",22,43], {"rightdomain": domainCoreStator, "leftdomain": 0, "maxh": hStator, "copy" : 4, "bc": minion5_bnd_label}],
               [["line",43,51], { "rightdomain": domainCoreStator, "leftdomain": 0, "maxh": hStator, "copy" : 5, "bc": minion6_bnd_label}],
               [["line",51,53], { "rightdomain": domainAirOut, "leftdomain": 0, "maxh": hAirOut, "copy" : 6, "bc": minion7_bnd_label}]]


    arcs = [[["spline3",0,4,3], {"leftdomain": 0, "rightdomain": domainAirInner, "maxh": hAirOut, "bc": shaft_bnd_label}],
            [["spline3",1,5,2], {"leftdomain": domainAirInner, "rightdomain": domainRotor, "maxh": hRotor}],
            [["spline3",6,8,7], {"leftdomain": domainRotor, "rightdomain": domainAirgapRotor, "maxh": hAirgap}],
            [["spline3",9,11,10], {"leftdomain": domainAirgapRotor, "rightdomain": domainAirgapStator, "maxh": hAirgap, "bc" : slidingBand_bnd_label}],
            [["spline3",50,54,51], {"leftdomain": domainCoreStator, "rightdomain": domainAirOut, "maxh": hAirOut}],
            [["spline3",52,55,53], {"leftdomain": domainAirOut, "rightdomain": 0, "maxh": hAirOut, "bc": out_bnd_label}],
            ]

    curvesAGshoes = [
            [["spline3",12,23,13], {"leftdomain": domainAirgapStator, "rightdomain": domainCoreStator, "maxh": hAirgap}],
            [["spline3",13,24,14], {"leftdomain": domainAirgapStator, "rightdomain": domainShoe11, "maxh": hAirgap}],
            [["spline3",14,25,15], {"leftdomain": domainAirgapStator, "rightdomain": domainShoe12, "maxh": hAirgap}],
            [["spline3",15,26,16], {"leftdomain": domainAirgapStator, "rightdomain": domainCoreStator, "maxh": hAirgap}],
            [["spline3",16,27,17], {"leftdomain": domainAirgapStator, "rightdomain": domainShoe21, "maxh": hAirgap}],
            [["spline3",17,28,18], {"leftdomain": domainAirgapStator, "rightdomain": domainShoe22, "maxh": hAirgap}],
            [["spline3",18,29,19], {"leftdomain": domainAirgapStator, "rightdomain": domainCoreStator, "maxh": hAirgap}],
            [["spline3",19,30,20], {"leftdomain": domainAirgapStator, "rightdomain": domainShoe31, "maxh": hAirgap}],
            [["spline3",20,31,21], {"leftdomain": domainAirgapStator, "rightdomain": domainShoe32, "maxh": hAirgap}],
            [["spline3",21,32,22], {"leftdomain": domainAirgapStator, "rightdomain": domainCoreStator, "maxh": hAirgap}],
            ]

    curvesShoes = [[["line",13,56], {"leftdomain": domainShoe11, "rightdomain": domainCoreStator, "maxh": hShoe}],
                [["line",14,57], {"leftdomain": domainShoe12, "rightdomain": domainShoe11, "maxh": hShoe}],
                [["line",15,58], {"leftdomain": domainCoreStator, "rightdomain": domainShoe12, "maxh": hShoe}],
                [["line",16,59], {"leftdomain": domainShoe21, "rightdomain": domainCoreStator, "maxh": hShoe}],
                [["line",17,60], {"leftdomain": domainShoe22, "rightdomain": domainShoe21, "maxh": hShoe}],
                [["line",18,61], {"leftdomain": domainCoreStator, "rightdomain": domainShoe22, "maxh": hShoe}],
                [["line",19,62], {"leftdomain": domainShoe31, "rightdomain": domainCoreStator, "maxh": hShoe}],
                [["line",20,63], {"leftdomain": domainShoe32, "rightdomain": domainShoe31, "maxh": hShoe}],
                [["line",21,64], {"leftdomain": domainCoreStator, "rightdomain": domainShoe32, "maxh": hShoe}],
                [["spline3",56,65,57], {"leftdomain": domainShoe11, "rightdomain": domainSlot11, "maxh": hSlot, "bc" : "testDirichlet"}],
                [["spline3",57,66,58], {"leftdomain": domainShoe12, "rightdomain": domainSlot12, "maxh": hSlot}],
                [["spline3",59,67,60], {"leftdomain": domainShoe21, "rightdomain": domainSlot21, "maxh": hSlot}],
                [["spline3",60,68,61], {"leftdomain": domainShoe22, "rightdomain": domainSlot22, "maxh": hSlot}],
                [["spline3",62,69,63], {"leftdomain": domainShoe31, "rightdomain": domainSlot31, "maxh": hSlot}],
                [["spline3",63,70,64], {"leftdomain": domainShoe32, "rightdomain": domainSlot32, "maxh": hSlot}],
            ]

    linesConductors = [[["line",56,34], {"leftdomain": domainSlot11, "rightdomain": domainCoreStator, "maxh": hStator}],
                [["line",57,35], {"leftdomain": domainSlot12, "rightdomain": domainSlot11, "maxh": hStator}],
                [["line",58,36], {"leftdomain": domainCoreStator, "rightdomain": domainSlot12, "maxh": hStator}],
                [["line",59,37], {"leftdomain": domainSlot21, "rightdomain": domainCoreStator, "maxh": hStator}],
                [["line",60,38], {"leftdomain": domainSlot22, "rightdomain": domainSlot21, "maxh": hStator}],
                [["line",61,39], {"leftdomain": domainCoreStator, "rightdomain": domainSlot22, "maxh": hStator}],
                [["line",62,40], {"leftdomain": domainSlot31, "rightdomain": domainCoreStator, "maxh": hStator}],
                [["line",63,41], {"leftdomain": domainSlot32, "rightdomain": domainSlot31, "maxh": hStator}],
                [["line",64,42], {"leftdomain": domainCoreStator, "rightdomain": domainSlot32, "maxh": hStator}]]

    curvesConductors = [[["spline3",34,44,35], {"leftdomain": domainSlot11, "rightdomain": domainCoreStator, "maxh": hStator}],
                [["spline3",35,45,36], {"leftdomain": domainSlot12, "rightdomain": domainCoreStator, "maxh": hStator}],
                [["spline3",37,46,38], {"leftdomain": domainSlot21, "rightdomain": domainCoreStator, "maxh": hStator}],
                [["spline3",38,47,39], {"leftdomain": domainSlot22, "rightdomain": domainCoreStator, "maxh": hStator}],
                [["spline3",40,48,41], {"leftdomain": domainSlot31, "rightdomain": domainCoreStator, "maxh": hStator}],
                [["spline3",41,49,42], {"leftdomain": domainSlot32, "rightdomain": domainCoreStator, "maxh": hStator}]]

    lines = periodic + arcs + curvesAGshoes + curvesShoes + linesConductors + curvesConductors
    
    ##################################################################################
    # Add bundles in the slots

    if bundles_per_half_slot >1:
    
        yminBundle = a/2
        ymaxBundle =  Rshoe * np.sin(thHalfSlot)
        ymin1 = yminBundle + ( (ymaxBundle+yminBundle)/2 - yminBundle) * (1-kBundleTh)
        ymax1 = ymaxBundle - ( (ymaxBundle+yminBundle)/2 - yminBundle) *(1-kBundleTh)

        dR =  kBundleR * (Ryoke - Rshoe) / bundles_per_half_slot
        counterDomain = domainAirOut
        lineBundle = []
        mat_bundles = {}
        R2 = Rshoe- (Ryoke - Rshoe) / bundles_per_half_slot * (1-kBundleR)/2
        for i_r in range(bundles_per_half_slot):
            #################################################################
            # Slot11
            counterPnt = len(pnts)-1
            counterDomain += 1
            R1 = R2 + (Ryoke - Rshoe) / bundles_per_half_slot * (1-kBundleR)
            R2 = R1 + dR
            th11 = np.asin(ymin1/R1)
            th12 = np.asin(ymax1/R1)
            th21 = np.asin(ymin1/R2)
            th22 = np.asin(ymax1/R2)

            pntBundle = [[(R1*np.cos(th11), R1*np.sin(th11)), {}],
                        [(R2*np.cos(th21), R2*np.sin(th21)), {}],
                        [(R2*np.cos(th22), R2*np.sin(th22)), {}],
                        [(R1*np.cos(th12), R1*np.sin(th12)), {}]]
            
            pntBundleSpline = [[find_tangent_intersection(pntBundle[1][0],pntBundle[2][0]), {}],
                            [find_tangent_intersection(pntBundle[0][0],pntBundle[3][0]), {}]]
            
            lineBundle += [[["line",counterPnt+1,counterPnt+2], {"leftdomain":counterDomain, "rightdomain": domainSlot11, "maxh": hBundle}],
                        [["spline3",counterPnt+2,counterPnt+5,counterPnt+3], {"leftdomain":counterDomain, "rightdomain": domainSlot11, "maxh": hBundle}],
                        [["line",counterPnt+3,counterPnt+4], {"leftdomain":counterDomain, "rightdomain": domainSlot11, "maxh": hBundle}],
                        [["spline3",counterPnt+4,counterPnt+6,counterPnt+1], {"leftdomain":counterDomain, "rightdomain": domainSlot11, "maxh": hBundle}]]
            
            mat_bundles[slot11_label + "_" + bundle_label + str(i_r)] = counterDomain

            pnts += pntBundle + pntBundleSpline
            #################################################################
            # Slot12
            counterPnt = len(pnts)-1
            counterDomain += 1

            th11 = thPole / 3 - np.asin(ymax1/R1)
            th12 = thPole / 3 - np.asin(ymin1/R1)
            th21 = thPole / 3 - np.asin(ymax1/R2)
            th22 = thPole / 3 - np.asin(ymin1/R2)

            pntBundle = [[(R1*np.cos(th11), R1*np.sin(th11)), {}],
                        [(R2*np.cos(th21), R2*np.sin(th21)), {}],
                        [(R2*np.cos(th22), R2*np.sin(th22)), {}],
                        [(R1*np.cos(th12), R1*np.sin(th12)), {}]]
            
            pntBundleSpline = [[find_tangent_intersection(pntBundle[1][0],pntBundle[2][0]), {}],
                            [find_tangent_intersection(pntBundle[0][0],pntBundle[3][0]), {}]]
            
            lineBundle += [[["line",counterPnt+1,counterPnt+2], {"leftdomain":counterDomain, "rightdomain": domainSlot12, "maxh": hBundle}],
                        [["spline3",counterPnt+2,counterPnt+5,counterPnt+3], {"leftdomain":counterDomain, "rightdomain": domainSlot12, "maxh": hBundle}],
                        [["line",counterPnt+3,counterPnt+4], {"leftdomain":counterDomain, "rightdomain": domainSlot12, "maxh": hBundle}],
                        [["spline3",counterPnt+4,counterPnt+6,counterPnt+1], {"leftdomain":counterDomain, "rightdomain": domainSlot12, "maxh": hBundle}]]
            
            mat_bundles[slot12_label + "_" + bundle_label + str(i_r)] = counterDomain

            pnts += pntBundle + pntBundleSpline

            #################################################################
            # Slot21
            counterPnt = len(pnts)-1
            counterDomain += 1

            th11 = np.asin(ymin1/R1) + thPole / 3
            th12 = np.asin(ymax1/R1) + thPole / 3
            th21 = np.asin(ymin1/R2) + thPole / 3
            th22 = np.asin(ymax1/R2) + thPole / 3

            pntBundle = [[(R1*np.cos(th11), R1*np.sin(th11)), {}],
                        [(R2*np.cos(th21), R2*np.sin(th21)), {}],
                        [(R2*np.cos(th22), R2*np.sin(th22)), {}],
                        [(R1*np.cos(th12), R1*np.sin(th12)), {}]]
            
            pntBundleSpline = [[find_tangent_intersection(pntBundle[1][0],pntBundle[2][0]), {}],
                            [find_tangent_intersection(pntBundle[0][0],pntBundle[3][0]), {}]]
            
            lineBundle += [[["line",counterPnt+1,counterPnt+2], {"leftdomain":counterDomain, "rightdomain": domainSlot21, "maxh": hBundle}],
                        [["spline3",counterPnt+2,counterPnt+5,counterPnt+3], {"leftdomain":counterDomain, "rightdomain": domainSlot21, "maxh": hBundle}],
                        [["line",counterPnt+3,counterPnt+4], {"leftdomain":counterDomain, "rightdomain": domainSlot21, "maxh": hBundle}],
                        [["spline3",counterPnt+4,counterPnt+6,counterPnt+1], {"leftdomain":counterDomain, "rightdomain": domainSlot21, "maxh": hBundle}]]
            
            mat_bundles[slot21_label + "_" + bundle_label + str(i_r)] = counterDomain

            pnts += pntBundle + pntBundleSpline

            #################################################################
            # Slot21
            counterPnt = len(pnts)-1
            counterDomain += 1

            th11 = 2*thPole / 3 - np.asin(ymax1/R1)
            th12 = 2*thPole / 3 - np.asin(ymin1/R1)
            th21 = 2*thPole / 3 - np.asin(ymax1/R2)
            th22 = 2*thPole / 3 - np.asin(ymin1/R2)

            pntBundle = [[(R1*np.cos(th11), R1*np.sin(th11)), {}],
                        [(R2*np.cos(th21), R2*np.sin(th21)), {}],
                        [(R2*np.cos(th22), R2*np.sin(th22)), {}],
                        [(R1*np.cos(th12), R1*np.sin(th12)), {}]]
            
            pntBundleSpline = [[find_tangent_intersection(pntBundle[1][0],pntBundle[2][0]), {}],
                            [find_tangent_intersection(pntBundle[0][0],pntBundle[3][0]), {}]]
            
            lineBundle += [[["line",counterPnt+1,counterPnt+2], {"leftdomain":counterDomain, "rightdomain": domainSlot22, "maxh": hBundle}],
                        [["spline3",counterPnt+2,counterPnt+5,counterPnt+3], {"leftdomain":counterDomain, "rightdomain": domainSlot22, "maxh": hBundle}],
                        [["line",counterPnt+3,counterPnt+4], {"leftdomain":counterDomain, "rightdomain": domainSlot22, "maxh": hBundle}],
                        [["spline3",counterPnt+4,counterPnt+6,counterPnt+1], {"leftdomain":counterDomain, "rightdomain": domainSlot22, "maxh": hBundle}]]
            
            mat_bundles[slot22_label + "_" + bundle_label + str(i_r)] = counterDomain

            pnts += pntBundle + pntBundleSpline

            #################################################################
            # Slot31
            counterPnt = len(pnts)-1
            counterDomain += 1
            
            th11 = np.asin(ymin1/R1) + 2*thPole / 3
            th12 = np.asin(ymax1/R1) + 2*thPole / 3
            th21 = np.asin(ymin1/R2) + 2*thPole / 3
            th22 = np.asin(ymax1/R2) + 2*thPole / 3

            pntBundle = [[(R1*np.cos(th11), R1*np.sin(th11)), {}],
                        [(R2*np.cos(th21), R2*np.sin(th21)), {}],
                        [(R2*np.cos(th22), R2*np.sin(th22)), {}],
                        [(R1*np.cos(th12), R1*np.sin(th12)), {}]]
            
            pntBundleSpline = [[find_tangent_intersection(pntBundle[1][0],pntBundle[2][0]), {}],
                            [find_tangent_intersection(pntBundle[0][0],pntBundle[3][0]), {}]]
            
            lineBundle += [[["line",counterPnt+1,counterPnt+2], {"leftdomain":counterDomain, "rightdomain": domainSlot31, "maxh": hBundle}],
                        [["spline3",counterPnt+2,counterPnt+5,counterPnt+3], {"leftdomain":counterDomain, "rightdomain": domainSlot31, "maxh": hBundle}],
                        [["line",counterPnt+3,counterPnt+4], {"leftdomain":counterDomain, "rightdomain": domainSlot31, "maxh": hBundle}],
                        [["spline3",counterPnt+4,counterPnt+6,counterPnt+1], {"leftdomain":counterDomain, "rightdomain": domainSlot31, "maxh": hBundle}]]
            
            mat_bundles[slot31_label + "_" + bundle_label + str(i_r)] = counterDomain

            pnts += pntBundle + pntBundleSpline

            #################################################################
            # Slot21
            counterPnt = len(pnts)-1
            counterDomain += 1

            th11 = thPole - np.asin(ymax1/R1)
            th12 = thPole - np.asin(ymin1/R1)
            th21 = thPole - np.asin(ymax1/R2)
            th22 = thPole - np.asin(ymin1/R2)

            pntBundle = [[(R1*np.cos(th11), R1*np.sin(th11)), {}],
                        [(R2*np.cos(th21), R2*np.sin(th21)), {}],
                        [(R2*np.cos(th22), R2*np.sin(th22)), {}],
                        [(R1*np.cos(th12), R1*np.sin(th12)), {}]]
            
            pntBundleSpline = [[find_tangent_intersection(pntBundle[1][0],pntBundle[2][0]), {}],
                            [find_tangent_intersection(pntBundle[0][0],pntBundle[3][0]), {}]]
            
            lineBundle += [[["line",counterPnt+1,counterPnt+2], {"leftdomain":counterDomain, "rightdomain": domainSlot32, "maxh": hBundle}],
                        [["spline3",counterPnt+2,counterPnt+5,counterPnt+3], {"leftdomain":counterDomain, "rightdomain": domainSlot32, "maxh": hBundle}],
                        [["line",counterPnt+3,counterPnt+4], {"leftdomain":counterDomain, "rightdomain": domainSlot32, "maxh": hBundle}],
                        [["spline3",counterPnt+4,counterPnt+6,counterPnt+1], {"leftdomain":counterDomain, "rightdomain": domainSlot32, "maxh": hBundle}]]
            
            mat_bundles[slot32_label + "_" + bundle_label + str(i_r)] = counterDomain

            pnts += pntBundle + pntBundleSpline


        lines += lineBundle
        
    if debug:
        plt.figure()
        plot_lines(pnts, lines, fac = 0.0005)
        plot_points(pnts)
        plt.show()

    if debug:
        plt.figure()
        plot_points(pnts)
        plt.show()

    ##################################################################################
    # Add points and curves to the geometry

    geo = SplineGeometry()
    for pnt, props in pnts:
        geo.AppendPoint(*pnt, **props)

    for line, props in lines:
        geo.Append(line, **props)

    ##################################################################################
    # Setup the mesh size

    geo.SetDomainMaxH(domainAirgapStator, hAirgap)
    geo.SetDomainMaxH(domainAirgapRotor, hAirgap)

    geo.SetDomainMaxH(domainAirOut, hAirOut)
    geo.SetDomainMaxH(domainAirInner, hAirOut)

    geo.SetDomainMaxH(domainRotor, hRotor)
    geo.SetDomainMaxH(domainCoreStator, hStator)

    geo.SetDomainMaxH(domainSlot11, hSlot)
    geo.SetDomainMaxH(domainSlot12, hSlot)
    geo.SetDomainMaxH(domainSlot21, hSlot)
    geo.SetDomainMaxH(domainSlot22, hSlot)
    geo.SetDomainMaxH(domainSlot31, hSlot)
    geo.SetDomainMaxH(domainSlot32, hSlot)

    geo.SetDomainMaxH(domainShoe11, hShoe)
    geo.SetDomainMaxH(domainShoe12, hShoe)
    geo.SetDomainMaxH(domainShoe21, hShoe)
    geo.SetDomainMaxH(domainShoe22, hShoe)
    geo.SetDomainMaxH(domainShoe31, hShoe)
    geo.SetDomainMaxH(domainShoe32, hShoe)

    if bundles_per_half_slot > 1:
        for key in mat_bundles.keys():
            geo.SetDomainMaxH(mat_bundles[key], hBundle)

    ##################################################################################
    # Setup the domains names

    geo.SetMaterial(domainAirgapStator, airgapStator_label)
    geo.SetMaterial(domainAirgapRotor, airgapRotor_label)
    
    geo.SetMaterial(domainAirOut, airOuter_label)
    geo.SetMaterial(domainAirInner, airInner_label)

    geo.SetMaterial(domainCoreStator, coreStator_label)
    geo.SetMaterial(domainRotor, rotor_label)

    geo.SetMaterial(domainSlot11, slot11_label)
    geo.SetMaterial(domainSlot12, slot12_label)
    geo.SetMaterial(domainSlot21, slot21_label)
    geo.SetMaterial(domainSlot22, slot22_label)
    geo.SetMaterial(domainSlot31, slot31_label)
    geo.SetMaterial(domainSlot32, slot32_label)

    geo.SetMaterial(domainShoe11, shoe11_label)
    geo.SetMaterial(domainShoe12, shoe12_label)
    geo.SetMaterial(domainShoe21, shoe21_label)
    geo.SetMaterial(domainShoe22, shoe22_label)
    geo.SetMaterial(domainShoe31, shoe31_label)
    geo.SetMaterial(domainShoe32, shoe32_label)

    if bundles_per_half_slot > 1:
        for key in mat_bundles.keys():
            geo.SetMaterial(mat_bundles[key], key)
                          
    ##################################################################################
    # Create and return the mesh

    return ngs.Mesh(geo.GenerateMesh(maxh = max([hAirOut, hRotor, hBundle, hShoe, hAirgap, hStator, hCorner])))


#%% Tests

if __name__ == "__main__" : 
    print("Test for 3 pole pair (debug on)")
    mesh = mesh_machine(p=3, debug=True).Curve(3)
    
    from ngsolve import Integrate
    print("Compute areas of bundles")
    print(str([(label, Integrate(1, mesh.Materials(label))) for label in 
               ['slot11_bundle0', 'slot12_bundle.*', 'slot21_bundle.*', 'slot22_bundle.*', 'slot31_bundle.*', 'slot32_bundle.*']]))
    print("Success!")