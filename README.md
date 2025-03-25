# Pseudo-kinematic trajectory control and planning of tracked vehicles




Michele Focchi, Daniele Fontanelli, Davide Stocco, Luigi Palopoli

This repository is a reduced version of [Locosim](https://github.com/mfocchi/locosim) ([preprint](https://arxiv.org/abs/2305.02107)) and it is intended for reproducing simulations and experiments
presented in the manuscript: Pseudo-kinematic trajectory control and planning of tracked vehicles

To run the optimization part of the code a Matlab license is required. 

# Installing the code

To install natively the code follow these detailed installation [instructions](https://github.com/mfocchi/tracked_robot_simulator/tree/master/install_native.md). However, we strongly suggest to install a docker image to avoid  compatibility issues. To see how to install the docker image follow these [instructions](https://github.com/mfocchi/tracked_robot_simulator/tree/master/install_docker.md). 



# **Running the Code**  

### IDE Pycharm

We recommend to use an IDE to run and edit the Python files, like Pycharm community. To install it,  you just need to download and unzip the program:

https://download.jetbrains.com/Python/pycharm-community-2021.1.1.tar.gz

 and unzip it  *inside* the home directory. 

To be able to keep the plots **alive** at the end of the program and to have access to variables,  you need to "Edit Configurations..." and tick "Run with Python Console". Otherwise the plot will immediately close. 

### Terminal

To run from a terminal we  use the interactive option that allows  when you close the program have access to variables:

```
$ python3 -i $LOCOSIM_DIR/robot_control/base_controllers/tractor_simulator.py
```

to exit from Python3 console type CTRL+Z







### Open loop Simulation

Run file **tracked_robot/tracked_robot_simulation3d.py**



### Training regressors



### Closed loop Simulation



The file **tractor_simulator.py** has certain option flags that are summarized in the following table:

| Flag | Value |
| ---- | ----- |
|      |       |
|      |       |
|      |       |
|      |       |
|      |       |
|      |       |



### Matlab Planning

In the case of Planning flag TODO



# Tips and Tricks 

1) Some machines, do not have support for GPU. This means that if you run Gazebo Graphical User Interface (GUI) it can become very **slow**. A way to mitigate this is to avoid to start the  Gazebo GUI and only start the gzserver process that will compute the dynamics, you will keep the visualization in Rviz. This is referred to planners that employ BaseController or BaseControllerFixed classes. In the Python code where you start the simulator you need to pass this additional argument as follows:

```
additional_args = 'gui:=false'
p.startSimulator(..., additional_args =additional_args)
```

2) Another annoying point is the default timeout to kill Gazebo that is by default very long. You can change it (e.g. to 0.1s) by setting the  _TIMEOUT_SIGINT = 0.1 and _TIMEOUT_SIGTERM = 0.1:

```
sudo gedit /opt/ros/ROS_VERSION/lib/PYTHON_PREFIX/dist-packages/roslaunch/nodeprocess.py
```

 this will cause ROS to send a `kill` signal much sooner.

3) if you get this annoying warning: 

```
Warning: TF_REPEATED_DATA ignoring data with redundant timestamp for frame...
```

a dirty hack to fix it is to clone this repository in your workspace:

```
git clone --branch throttle-tf-repeated-data-error git@github.com:BadgerTechnologies/geometry2.git
```

 