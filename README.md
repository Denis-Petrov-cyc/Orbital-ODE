# ORBITAL DYNAMICS SIMULATOR
The Orbital Dynamics Simulator is a Python-based 3D n-body orbital simulator using Newtonian gravitation and numerical ODE integration.

## FEATURES
- Simulates gravitational interactions between multiple celestial bodies
- Implements fourth-order Runge-Kutta (RK4) and Euler integration
- Visualizes orbital trajectories in 3D
- Tracks inter-body distances and total mechanical energy
- Compares numerical accuracy through energy conservation

## MODEL
Each body's acceleration is calculated from the gravitational influence of every other body:

$$
\mathbf{a}_i = G \sum_{j \ne i} m_j \frac{\mathbf{r}_j-\mathbf{r}_i}{\lVert\mathbf{r}_j-\mathbf{r}_i\rVert^3}
$$

The system is integrated numerically to update each body's position and velocity over time.

## Built With
Python / NumPy / SciPy / Matplotlib

## Run
```bash
pip install numpy scipy matplotlib
python orbitalDynamics.py
```
