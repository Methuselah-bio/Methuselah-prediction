"""
ode_simulation.py
-----------------

This utility script simulates a simple nutrient perturbation model to
generate synthetic data for training or benchmarking.  While real
chronological lifespan (CLS) depends on complex regulatory pathways
such as TOR signalling and autophagy, we approximate nutrient
dynamics using a basic ordinary differential equation (ODE) system:

```
dN/dt = -k * C * N           # nutrient consumption proportional to cell density
dC/dt = r * C * (1 - C/K)    # logistic growth of cell population
```

where N is the nutrient concentration, C is the cell population, k is
the consumption rate, r is the growth rate and K is the carrying
capacity.  The model generates trajectories that can be used as
features or targets in machine‑learning models.  If ``scipy`` is not
available, the script performs a simple Euler integration.

Usage:

```bash
python src/ode_simulation.py --output results/simulated_data.csv --t-final 10 --dt 0.1
```

"""

import argparse
import logging
import os
import numpy as np
import pandas as pd

try:
    from scipy.integrate import solve_ivp  # type: ignore
except ImportError:
    solve_ivp = None  # type: ignore


def simulate_ode(t_final: float = 10.0, dt: float = 0.1, k: float = 0.05, r: float = 0.3, K: float = 1.0,
                 N0: float = 1.0, C0: float = 0.1) -> pd.DataFrame:
    """Simulate nutrient and cell population dynamics.

    Parameters
    ----------
    t_final : float
        Final time of simulation.
    dt : float
        Time step for integration (used in Euler method if scipy is not
        available).
    k : float
        Nutrient consumption rate.
    r : float
        Growth rate of cell population.
    K : float
        Carrying capacity.
    N0 : float
        Initial nutrient concentration.
    C0 : float
        Initial cell population.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['time', 'nutrient', 'cells'].
    """
    def odes(t, y):
        N, C = y
        dN_dt = -k * C * N
        dC_dt = r * C * (1 - C / K)
        return [dN_dt, dC_dt]

    if solve_ivp is not None:
        t_span = (0, t_final)
        t_eval = np.arange(0, t_final + dt, dt)
        sol = solve_ivp(odes, t_span, [N0, C0], t_eval=t_eval, vectorized=False)
        times = sol.t
        N_vals = sol.y[0]
        C_vals = sol.y[1]
    else:
        # Simple Euler integration
        n_steps = int(t_final / dt) + 1
        times = np.linspace(0, t_final, n_steps)
        N_vals = [N0]
        C_vals = [C0]
        for i in range(1, n_steps):
            N_prev, C_prev = N_vals[-1], C_vals[-1]
            dN, dC = odes(times[i - 1], [N_prev, C_prev])
            N_vals.append(N_prev + dt * dN)
            C_vals.append(C_prev + dt * dC)
    df = pd.DataFrame({'time': times, 'nutrient': N_vals, 'cells': C_vals})
    return df


def main():
    parser = argparse.ArgumentParser(description='Simulate nutrient and cell dynamics.')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file for simulated data.')
    parser.add_argument('--t-final', type=float, default=10.0, help='Final time of simulation.')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step for integration.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    df = simulate_ode(t_final=args.t_final, dt=args.dt)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output, index=False)
    logging.info(f"Simulated data saved to {args.output}")


if __name__ == '__main__':
    main()