"""
import numpy as np 
import skfuzzy as fuzz 
from skfuzzy import control as ctrl 
import matplotlib.pyplot as plt 
 
TempError = ctrl.Antecedent(np.arange(-5, 6, 1), 'TempError') 
Humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'Humidity') 
CompressorSpeed = ctrl.Consequent(np.arange(0, 101, 1), 'CompressorSpeed') 
 
TempError['Cold'] = fuzz.trimf(TempError.universe, [-5, -5, 0]) 
TempError['Good'] = fuzz.trimf(TempError.universe, [-1, 0, 1]) 
TempError['Hot'] = fuzz.trimf(TempError.universe, [0, 5, 5]) 
 
Humidity['Low'] = fuzz.trimf(Humidity.universe, [0, 0, 50]) 
Humidity['High'] = fuzz.trimf(Humidity.universe, [50, 100, 100]) 
 
CompressorSpeed['Off'] = fuzz.trimf(CompressorSpeed.universe, [0, 0, 10]) 
CompressorSpeed['Slow'] = fuzz.trimf(CompressorSpeed.universe, [10, 30, 50]) 
CompressorSpeed['Medium'] = fuzz.trimf(CompressorSpeed.universe, [40, 60, 80]) 
CompressorSpeed['Fast'] = fuzz.trimf(CompressorSpeed.universe, [70, 100, 100]) 
 
rule1 = ctrl.Rule(TempError['Cold'], CompressorSpeed['Off']) 
rule2 = ctrl.Rule(TempError['Good'], CompressorSpeed['Slow']) 
rule3 = ctrl.Rule(TempError['Hot'] & Humidity['Low'], CompressorSpeed['Medium']) 
rule4 = ctrl.Rule(TempError['Hot'] & Humidity['High'], CompressorSpeed['Fast']) 
 
ac_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4]) 
ac_sim = ctrl.ControlSystemSimulation(ac_ctrl) 
# ...existing code...
if __name__ == "__main__":
    print("\n--- Evaluating Controller ---") 
 
    ac_sim.input['TempError'] = 3 
    ac_sim.input['Humidity'] = 30 
    ac_sim.compute() 
    print(f"Case 1 (Hot, Low Humidity): Compressor Speed = {ac_sim.output['CompressorSpeed']:.2f}%") 
 
    ac_sim.input['TempError'] = 3 
    ac_sim.input['Humidity'] = 80 
    ac_sim.compute() 
    print(f"Case 2 (Hot, High Humidity): Compressor Speed = {ac_sim.output['CompressorSpeed']:.2f}%") 
 
    ac_sim.input['TempError'] = -2 
    ac_sim.input['Humidity'] = 50 
    ac_sim.compute() 
    print(f"Case 3 (Too Cold): Compressor Speed = {ac_sim.output['CompressorSpeed']:.2f}%") 
 
    fig, axs = plt.subplots(2, 2, figsize=(10, 7)) 
 
    TempError.view(ax=axs[0, 0]) 
    axs[0, 0].set_title('Temperature Error Membership Functions') 
 
    Humidity.view(ax=axs[0, 1]) 
    axs[0, 1].set_title('Humidity Membership Functions') 
 
    # show compressor membership functions and mark last computed output
    CompressorSpeed.view(ax=axs[1, 0]) 
    axs[1, 0].set_title('Compressor Speed Membership Functions') 
    # mark the last computed crisp output on the membership plot (Case 3)
    try:
        axs[1, 0].axvline(ac_sim.output['CompressorSpeed'], color='r', linestyle='--', label='Output')
        axs[1, 0].legend()
    except Exception:
        pass
 
    # hide unused subplot
    axs[1, 1].axis('off')
 
    plt.tight_layout() 
    plt.show()
"""

import random# ...existing code...
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