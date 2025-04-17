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

# Closed loop simulation

To run from a terminal we  use the interactive option that allows  when you close the program have access to variables:

```
$ python3 -i $LOCOSIM_DIR/robot_control/base_controllers/closed_loop_simulation.py
```

to exit from Python3 console type CTRL+Z

The file **closed_loop_simulation.py** has certain option flags that are summarized in the following table:

| Flag                   | Value                                                    | Meaning                                                      |
| ---------------------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| SIMULATOR              | 'distributed2d','distributed3d'                          | set to flat_terran/sloped terrain                            |
| ControlType            | 'OPEN_LOOP','CLOSED_LOOP_UNICYCLE', 'CLOSED_LOOP_SLIP_0' | set to CLOSED_LOOP_UNICYCLE for UC controller, and to CLOSED_LOOP_SLIP_0 for SLC controller |
| SIDE_SLIP_COMPENSATION | 'NONE','MACHINE_LEARNING'                                |                                                              |
| LONG_SLIP_COMPENSATION | 'NONE','NN'                                              |                                                              |
| IDENT_TYPE             | 'NONE','WHEELS'                                          | set to WHEELS to perform the open-loop tests for regressor identification |
| STATISTICAL_ANALYSIS   | False, True                                              | set to WHEELS to perform the statistical analisys            |
| friction_coefficient   | 0,1, 0.4, 0.6                                            |                                                              |
| PLANNING               | 'none', dubins' , 'clothoids', 'optim'                   |                                                              |
| SAVE_BAGS              | False, True                                              |                                                              |
| ADD_NOISE              | False, True                                              |                                                              |

### Regressor Identification on flat terrain

The slippage regressor will be used if ControlType=CLOSED_LOOP_SLIP_0  with  SIDE_SLIP_COMPENSATION=MACHINE_LEARNING and LONG_SLIP_COMPENSATION=MACHINE_LEARNING. To run the identification tests on flat terrain:

1) set in the file **closed_loop_simulation.py** 

   ```
   IDENT_TYPE='WHEELS', 
   SIMULATOR='distributed2d'
   IDENT_TYPE='WHEELS'
   ControlType='OPEN_LOOP'
   friction_coefficient = 0.1/0.4
   ```


2. run the simulation, many .csv files will be generated  in the folder **robot_control/tracked_robot/regressor/data2d**
3. run **generate_slippage_regressor2d.py**, set the friction coefficient to either 0.1 and 0.4. This command will generate the catboost model files **model_alpha0.1.cb, model_beta_l0.1.cb, ** **model_beta_r0.1.cb** and **model_alpha0.4.cb, model_beta_l0.4.cb, ** **model_beta_r0.4.cb**



### Regressor Identification on sloped terrain

The slippage regressor will be used if ControlType=CLOSED_LOOP_SLIP_0  with  SIDE_SLIP_COMPENSATION=MACHINE_LEARNING and LONG_SLIP_COMPENSATION=MACHINE_LEARNING. To run the identification tests on flat terrain:

1) set in the file **closed_loop_simulation.py** 

   ```
   SIMULATOR='distributed3d'
   IDENT_TYPE='WHEELS'
   ControlType='OPEN_LOOP'
   friction_coefficient = 0.4/0.6
   ```

2. run the simulation, many .csv files will be generated  in the folder **robot_control/tracked_robot/regressor/data2d**
3. run **generate_slippage_regressor3d.py**, set the friction coefficient to either 0.4 and 0.6. This command will generate the catboost model files **model_alpha_3d_0.4.cb, model_beta_l_3d_0.4.cb, ** **model_beta_r_3d_0.4.cb** and **model_alpha_3d_0.6.cb, model_beta_l_3d_0.6.cb, ** **model_beta_r_3d_0.6.cb**



## **Experiments on Flat terrain **

### **User defined reference**

To generate Figure 11,12, 13 

1. set in the file **closed_loop_simulation.py** 

```
IDENT_TYPE='NONE', 
SIMULATOR='distributed2d'
ControlType='CLOSED_LOOP_SLIP_0' / 'CLOSED_LOOP_UNICYCLE'
friction_coefficient = 0.1
ADD_NOISE=True
p0 = np.array([0.05, 0.03, 0.01])
des_x = 0
des_y = 0
des_theta = 0
```

2. run **closed_loop_simulation.py**

### **Planning strategies evaluation**

To generate Figure 15, 16

1. set in the file  **closed_loop_simulation.py** 

```
IDENT_TYPE='NONE', 
SIMULATOR='distributed2d'
ControlType='CLOSED_LOOP_SLIP_0' / 'CLOSED_LOOP_UNICYCLE'
friction_coefficient = 0.4
self.PLANNING =  'dubins' /'optim'/ 'clothoids'
ADD_NOISE=False
```

2. run matlab__optimization/ros/ros_node.m
3. run closed_loop_simulation.py

### **Statistical analisys**

To generate Table 4:

1. set in the file  **closed_loop_simulation.py** 

```
IDENT_TYPE='NONE', 
SIMULATOR='distributed2d'
ControlType='CLOSED_LOOP_SLIP_0' / 'CLOSED_LOOP_UNICYCLE'
friction_coefficient = 0.4
self.PLANNING =  'dubins' /'optim'/ 'clothoids'
ADD_NOISE=False
STATISTICAL_ANALYSIS = True
```

2. run **matlab_optimization/ros/ros_node.m**

3. run **closed_loop_simulation.py**, the data will be stored in **statistics.csv**



## **Experiments on slopes**

### Open loop Simulation 

To generate Figure 20 (left):

1. in **open_loop_simulation3d.py** uncomment what is related to "PAPER: sloped test: distributed/non distributed"

2. set contact_distribution=True/False
3. run **open_loop_simulation3d.py**

4.  a ****openLoop3DModel_sloped_test_chicane_Distr_{True/False}.mat**** file is saved with the log of the  pose variable inside the robot_control/base_controller/tracked_robot/simulator folder

To generate Figure 20 (right)

1. in **open_loop_simulation3d.py** uncomment what is related to "PAPER: Stiffness variation"

2. set contact_distribution=True/False
3. run **open_loop_simulation3d.py**
4. a **openLoop3DModel_stiff_var_test_chicane_Distr_{True/False}.mat** file is saved with the log of the  pose variable inside the robot_control/base_controller/tracked_robot/simulator folder

### **Closed loop simulation**

1. In the file **closed_loop_simulation.py** set:

```
SIMULATOR='distributed3d'
TERRAIN = True
ControlType='CLOSED_LOOP_SLIP_0' / 'CLOSED_LOOP_UNICYCLE'
friction_coefficient = 0.6
IDENT_TYPE='NONE'
PLANNING =  'none'
ADD_NOISE=False
STATISTICAL_ANALYSIS = False
```

2. run **closed_loop_simulation.py** 



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

 