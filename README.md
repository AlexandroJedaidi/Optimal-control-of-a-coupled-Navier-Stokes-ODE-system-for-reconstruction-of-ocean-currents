# Optimal-control-of-a-coupled-Navier-Stokes-ODE-system-for-reconstruction-of-ocean-currents
Optimal control of a coupled Navier-Stokes ODE system for reconstruction of ocean currents

First make sure, Fenics is installed, see https://fenicsproject.org/download/archive/

Additional requirements:
- numpy
- matplotlib
- mshr

To run an experiment, choose between the files OCP_dolfin.py, Pipeline_limits.py, initial_control_test.py

1. set up the parameters.json
2. set up the pipeline in the first lines 
3. run python3 PIPELINE_NAME.py

General Settings of pipelines:
- experiment: used to create new directories to not overwrite old runs
- ud_experiment: change specific run of experiment, options found in comment in the same line. If none are listed, do not change (that is only the case in initial_control_test.py).
- num_steps: max. steps of Gradient descent
- np_path: location of result directory
- x_resolution: configures the mesh resolution
- grad_check: turns Gradient check for the run on/off
- use_line_search: turns line search on/off
- line search setup: condition threshold, decreasing rate and max/min learning rate can be modified
- conv_crit: convergence criterion can be adjusted

For OCP_dolfin.py, there are two modes:
1. mode: L_shape=False, there we run the OCP on a square mesh just as described in the thesis
2. mode: L_shape=True, there we run the OCP on L-shape mesh, as the second experiment in the thesis describes

For initial_control_test.py, one can choose between multiple initial controls, as stated in thesis.
For changing the control, change case=X, where X=0,1,2,3