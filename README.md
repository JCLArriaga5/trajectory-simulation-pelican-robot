# Pelican Experimental Robot Trajectory Simulation

<p align="center"><img src="images/pelican_prototype.jpeg"></p>

![License](https://img.shields.io/github/license/JCLArriaga5/trajectory-simulation-pelican-robot) ![size](https://img.shields.io/github/repo-size/JCLArriaga5/trajectory-simulation-pelican-robot)

Trajectory simulation of the Pelican Prototype Robot of the [CICESE Research Center, Mexico](https://www.cicese.edu.mx/). The position control used was PD control with gravity compensation and the 4th order Runge-Kutta method.

## Use the repository
```sh
$ git clone https://github.com/JCLArriaga5/trajectory-simulation-pelican-robot.git
```
## Simulation
The simulation was carried out with a Colab notebook using the functions created in [pelcnrbt](../master/pelcnrbt/), the simulation notebook can be seen [here](../master/simulation_nb/pelican_trajectory_simulation.ipynb).

### Images for desired position (0.26, 0.13)
**Graph of error**: The graph shows how the error of the angles for the desired position was reduced by the controller. The errors shown in the graph are a product of the friction phenomenon that has not been included in the dynamics of the robot.

```python
sim.plot_q_error()
```
<p align="center"><img src="images/error_graph.png"></p>

**Graph of velocities behavior**: The graph shows how the speed of each link behaved until reaching the desired position
```python
sim.plot_velocity_bhvr()
```
<p align="center"><img src="images/qps_bhvr_graph.png"></p>

**Graph of desired position trajectory**
```python
sim.plot_trajectory(50)
```
<p align="center"><img src="images/trajectory_graph.png"></p>

### Trajectory Animation
<p align="center"><img src="images/pelican_trajectory_anim.gif"></p>

# References
 Kelly, R., Davila, V. S., & Perez, J. A. L. (2006). Control of robot manipulators in joint space. Springer Science & Business Media.
