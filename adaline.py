# ...existing code...
import numpy as np
import math
# ...existing code...

def specific_rotation(alpha: float, l_cm: float, weight_g: float, volume_ml: float) -> float:
    """
    Calculate specific rotation [α] of a solution.

    Parameters
    ----------
    alpha : float
        Observed rotation in degrees.
    l_cm : float
        Polarimeter tube length in centimeters.
    weight_g : float
        Mass of solute (grams) dissolved in the solution.
    volume_ml : float
        Volume of the prepared solution in millilitres.

    Returns
    -------
    float
        Specific rotation [α] in degrees · dm⁻¹ · (g/100mL)⁻¹

    Raises
    ------
    ValueError
        If any inputs are non-finite, l_cm <= 0, volume_ml <= 0, or weight_g < 0.
    """
    # Validate numeric finiteness
    if not all(math.isfinite(x) for x in (alpha, l_cm, weight_g, volume_ml)):
        raise ValueError("All inputs must be finite numbers")

    # Basic validation
    if l_cm <= 0:
        raise ValueError("Tube length (l_cm) must be > 0 cm")
    if volume_ml <= 0:
        raise ValueError("Volume (volume_ml) must be > 0 mL")
    if weight_g < 0:
        raise ValueError("Weight (weight_g) must be >= 0 g")

    # Convert path length to dm
    l_dm = l_cm / 10.0

    # Concentration in g per 100 mL
    p = (weight_g / volume_ml) * 100.0

    if p <= 0:
        raise ValueError("Concentration must be > 0 (no solute or invalid inputs)")

    # Formula: [α] = 100 * α / (l * p)
    specific_alpha = (100.0 * alpha) / (l_dm * p)
    return specific_alpha
# ...existing code...

if __name__ == "__main__":
    # Improved CLI-driven example + nicer output formatting
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="specific_rotation",
        description="Compute specific rotation [α] for a solution (° · dm⁻¹ · (g/100mL)⁻¹)."
    )
    parser.add_argument("--alpha", type=float, default=13.2, help="Observed rotation in degrees (default: 13.2)")
    parser.add_argument("--length-cm", type=float, dest="l_cm", default=20.0, help="Tube length in cm (default: 20.0)")
    parser.add_argument("--weight-g", type=float, dest="weight_g", default=2.0, help="Mass of solute in g (default: 2.0)")
    parser.add_argument("--volume-ml", type=float, dest="volume_ml", default=100.0, help="Volume of solution in mL (default: 100.0)")
    args = parser.parse_args()

    # Validate parsed values before computation
    try:
        if not all(math.isfinite(x) for x in (args.alpha, args.l_cm, args.weight_g, args.volume_ml)):
            raise ValueError("All inputs must be finite numbers")
        result = specific_rotation(args.alpha, args.l_cm, args.weight_g, args.volume_ml)
    except ValueError as exc:
        print("Error:", exc, file=sys.stderr)
        sys.exit(2)
    else:
        conc = (args.weight_g / args.volume_ml) * 100.0
        print("Specific Rotation Calculation")
        print("-----------------------------")
        print(f"Observed rotation (α): {args.alpha:.6f} °")
        print(f"Tube length (l):      {args.l_cm:.3f} cm")
        print(f"Mass of solute:       {args.weight_g:.6f} g")
        print(f"Volume of solution:   {args.volume_ml:.6f} mL")
        print(f"Concentration (p):    {conc:.6f} g/100mL")
        print(f"\nSpecific Rotation [α]: {result:.6f} ° · dm⁻¹ · (g/100mL)⁻¹")
# ...existing code...