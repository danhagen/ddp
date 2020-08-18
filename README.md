# ddp
## Differential Dynamic Programming!... and Model Predictive Control.

This repository utilizes differential dynamic programming (and in some special cases, model predictive control) to control examples of dynamical systems like the simple pendulum or the more complicated cart-pendulum. Foundational work was also conducted to control the position of a pendulum antagonistically-actuated by 'muscle'-like actuators and compliant tendons.

To clone this repository, simply navigate to your desired folder and from the command line type,
```
git clone https://github.com/danhagen/ddp.git && cd ddp
```

## Simple Pendulum
From a `python` interface type
```py
run ddp_simple_pendulum.py
```
or from the command line type
```
python ddp_simple_pendulum.py
```
in order to run the toy problem of controlling the position of a simple pendulum with differential dynamic programming. The program can be animated by adding the option `--animate` (sample animation below).

<p align="center">
  <img width="500" src="https://raw.githubusercontent.com/danhagen/ddp/master/visualizations_simple_pendulum/simple_pendulum_ddp.gif" alt="Simple Pendulum -- DDP"></br>
  <small>Fig. 1: Controlling a simple pendulum with DDP.</small> 
</p>

## Cart-Pendulum
From a `python` interface type
```py
run ddp_cart_pendulum_example.py
```
or from the command line type
```
python ddp_cart_pendulum_example.py
```
in order to run the toy problem of controlling the position of a pendulum situated on top of a cart with differential dynamic programming. The program can be animated by adding the option `--animate` (sample animation below).

<p align="center">
  <img width="500" src="https://raw.githubusercontent.com/danhagen/ddp/master/visualizations_cart_pendulum/cart_pendulum_ddp.gif" alt="Cart-Pendulum -- DDP"></br>
  <small>Fig. 2: Controlling pendulum angle by pushing the cart on which it is situated with DDP.</small> 
</p>


## Pendulum w/ Muscles (1 DOF, 2 DOA) 

Alternatively, the position of a pendulum that is actuated by two 'muscle'-like actuators that pull on tendons with nonlinear elasticity can be controlled via either DDP or model predictive control (MPC) using the the subdirectory `ddp/1DOF2DOA/` and executing the following commands from a `python` interface.

```py
run run_1DOF_1DOA_Torq_DDP.py # DDP controller w/ angular torque as input (1DOF)
run run_1DOF_1DOA_Torq_MPC.py # MPC controller w/ angular torque as input (1DOF)
run run_1DOF_2DOA_TT_DDP.py # DDP controller with tendon tensions as inputs (2DOFs)
run run_1DOF_2DOA_TT_MPC.py # MPC controller with tendon tensions as inputs (2DOFs)
```

Sample animations are provided below for tendon-tension controlled examples. 

<p align="center">
  <img width="500" src="https://raw.githubusercontent.com/danhagen/ddp/master/1DOF_2DOA/visualizations/1DOF_2DOA_TT/DDP/2019_06_05_170215/1DOF_2DOA_TT_DDP.gif" alt="Pendulum with Two Muscles (Tension-controlled DDP)"></br>
  <small>Fig. 3: DDP controller with tendon tensions as inputs (2DOFs).</small> 
</p>

<p align="center">
  <img width="500" src="https://raw.githubusercontent.com/danhagen/ddp/master/1DOF_2DOA/visualizations/1DOF_2DOA_TT/MPC/2019_05_08_101623/1DOF_2DOA_TT_MPC.gif" alt="Pendulum with Two Muscles (Tension-controlled MPC)"></br>
  <small>Fig. 4: MPC controller with tendon tensions as inputs (2DOFs).</small> 
</p>
