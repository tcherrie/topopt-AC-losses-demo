"""Supply utilities

Provide utilities related to the electric supply of the machine.

Functions defined here:
- phase_current
- winding_arrangement
- bundle_arrangement

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

#%% Current feeding

def phase_current(I_rms : float,
                  load_angle : float=0,
                  Ap : str = "Ap",
                  Am : str = "Am",
                  Bp : str = "Bp",
                  Bm : str = "Bm",
                  Cp : str = "Cp",
                  Cm : str = "Cm"
                  ) -> dict:
    """
    Generate a balanced three-phase current system from RMS magnitude.

    This function constructs complex phase currents for a symmetrical
    three-phase system using a specified RMS current and electrical
    load angle.

    Parameters
    ----------
    I_rms : float
        RMS value of the phase current.

    load_angle : float, optional
        Electrical phase reference angle (rad).

    Ap, Am, Bp, Bm, Cp, Cm : str, optional
        Labels for the positive and negative conductors of each phase
        (A, B, C). These are used as keys in the returned dictionary.

    Returns
    -------
    dict
        Dictionary mapping conductor labels to complex phase currents.

    Notes
    -----
    - The system is assumed balanced and sinusoidal.
    - Phase shifts follow:
        A → 0
        B → -2π/3
        C → -4π/3
    - Negative conductors carry the opposite current of their positive
      counterparts.
    """

    phase = {
        Ap: ngs.sqrt(2) * I_rms * ngs.exp(1j * load_angle),
        Am: -ngs.sqrt(2) * I_rms * ngs.exp(1j * load_angle),
        Bp: ngs.sqrt(2) * I_rms * ngs.exp(1j * (load_angle - 2 * ngs.pi / 3)),
        Bm: -ngs.sqrt(2) * I_rms * ngs.exp(1j * (load_angle - 2 * ngs.pi / 3)),
        Cp: ngs.sqrt(2) * I_rms * ngs.exp(1j * (load_angle - 4 * ngs.pi / 3)),
        Cm: -ngs.sqrt(2) * I_rms * ngs.exp(1j * (load_angle - 4 * ngs.pi / 3)),
    }
    return phase

#%% Arrangement

def winding_arrangement(phase : dict,
                        type : str,
                        slot11 : str = "slot11",
                        slot12 : str = "slot12",
                        slot21 : str = "slot21",
                        slot22 : str = "slot22",
                        slot31 : str = "slot31",
                        slot32 : str = "slot32",
                        Ap : str = "Ap",
                        Am : str = "Am",
                        Bp : str = "Bp",
                        Bm : str = "Bm",
                        Cp : str = "Cp",
                        Cm : str = "Cm"
                        ) -> dict:
    """
    Map phase currents to stator slot positions.

    This function defines the spatial distribution of phase currents in the
    machine slots based on a выбран winding configuration (distributed or
    concentrated).

    Parameters
    ----------
    phase : dict
        Dictionary of phase currents as returned by `phase_current`.

    type : str
        Winding topology: "distributed", "concentrated"

    slot11, slot12, ..., slot32 : str, optional
        Slot identifiers used as keys in the returned dictionary.

    Ap, Am, Bp, Bm, Cp, Cm : str, optional
        Phase conductor labels used to index the `phase` dictionary.

    Returns
    -------
    dict
        Dictionary mapping slot names to complex currents.

    Notes
    -----
    - This function defines only the spatial assignment of currents.
    - The actual electrical phase definition must be provided via
      `phase_current`.
    """
    
    if type.lower() == "distributed":
        winding = {
            slot11: phase[Ap],
            slot12: phase[Ap],
            slot21: phase[Cm],
            slot22: phase[Cm],
            slot31: phase[Bp],
            slot32: phase[Bp],
        }

    elif type.lower() == "concentrated":
        winding = {
            slot11: phase[Ap],
            slot12: phase[Cp],
            slot21: phase[Cm],
            slot22: phase[Bm],
            slot31: phase[Bp],
            slot32: phase[Ap],
        }

    return winding


def bundle_arrangement(winding: dict,   # comes from winding_arrangement
                       bundles_per_half_slot: int,
                       bundle_label: str = "bundle"
                       ) -> dict:
    """
    Expand slot currents into individual conductor bundles.

    This function distributes the slot-level winding currents into multiple
    parallel conductor bundles per half-slot.

    Parameters
    ----------
    winding : dict
        Dictionary mapping slot names to complex currents, typically
        produced by `winding_arrangement`.

    bundles_per_half_slot : int, optional
        Number of conductor bundles per half-slot.

    bundle_label : str, optional
        Label prefix used to distinguish bundles in the returned keys.

    Returns
    -------
    dict
        Dictionary mapping bundle identifiers to complex currents.

    Notes
    -----
    - Each slot current is replicated across all bundles in that slot.
    - The resulting keys follow the pattern:
        "<slot>_<bundle_label><index>"
    - This function does not redistribute current magnitude; it only
      duplicates the assignment.
    """

    bundles = {}

    # Loop over bundle subdivisions
    for i in range(bundles_per_half_slot):

        # Assign each slot current to all bundles in that slot
        for key in winding.keys():
            bundles[key + "_" + bundle_label + str(i)] = winding[key]

    return bundles
