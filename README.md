# iris_robots: Franka
## Adaptation of polymetis for Franka 

## Setup instructions
If working on a workstation / separate machine from controller (NUC), make a conda env using `conda create -n polymetis python=3.8`

Install polymetis as `conda install -c pytorch -c fair-robotics -c aihabitat -c conda-forge polymetis`
(TODO: add instructions for polymetis-local + changes to enable server)
Note: This will take a long time to populate (few minutes), be patient.

Need to also install dm_control (latest verison) and dm_robotics package to
use the IK solver

``pip install dm-robotics-moma``

Also when installing on NUC, conda commands don't work. Go through local server install process.

Additional packages also needed (TODO: put these in a yaml file):
gym
OpenCV
pygame
ipdb
absl

After installation, add the directory to the PYTHONPATH:

```
export PYTHONPATH=$PYTHONPATH:~/iris_robots/
```

## Using the interface
The RobotEnv is a OpenAI gym-style interface with the polymetis stack under the hood.
If running on NUC itself, FrankaRobot will be instantiated. Otherwise, the RobotInterface
will be used. The documentation assumes the use of `polymetis-local`.

### Step 1: Start a server.
First, start a server to communicate with the robot
on the NUC. Note, this requires a local installation of polymetis.

```
tmux new-session
sudo pkill -9 run_server
conda activate polymetis-local
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=172.16.0.2
```

If doing an overnight run, start a persistent server:
```
tmux new-session
sudo pkill -9 run_server
conda activate polymetis-local
bash ~/fairo/polymetis/polymetis/python/scripts/persist_server.sh
```

### Step 2: Start the gripper.
Then, create [a new window in tmux](https://tmuxcheatsheet.com/)
and activate the gripper. The terminal should prompt that the 
robotiq gripper is activated. If you get an error, try again :)
```
sudo chmod a+rw /dev/ttyUSB0
conda activate polymetis-local
launch_gripper.py gripper=robotiq_2f gripper.comport=/dev/ttyUSB0
```

### (Required for training) Step 3: Start the flask server.
Create a new window and run the flask server, which communicates between
the nuc / workstation. NOTE: this is needed only if you're training the models on a separate machine (for example, workstation).
```
conda activate polymetis-local
python ~/iris_robots/run_server.py
```

# Notes on Server
Running the launch robot command uses the yaml file at
`/home/panda5/fairo/polymetis/polymetis/conf/launch_robot.yaml`
Also gives different options for choice of robot client.

The file that determines the number of allowed automatic error recovery
attempts is at
` fairo/polymetis/polymetis/src/clients/franka_panda_client/franka_panda_client.cpp `
And the constants used for all of this are here
 `fairo/polymetis/polymetis/include/polymetis/clients/franka_panda_client.hpp` 

 As for the server warning themselves, the warning for exceed the threshold of 
 force is logged here
`fairo/polymetis/polymetis/src/polymetis_server.cpp`


# Demo Collection
The repo should already have the module/submodule structure created
so run the demo recording script as a module
```
python -m iris_robots.record_demos
```

# Error FAQ
If when running the server, you see the following or similar error
```
dm_control/mjcf/physics.py", line 495, in from_mjcf_model
    return cls.from_xml_string(xml_string=xml_string, assets=assets)
  File "/home/panda5/anaconda3/envs/polymetis-local/lib/python3.8/site-packages/dm_control/mujoco/engine.py", line 424, in from_xml_string
    return cls.from_model(model)
  File "/home/panda5/anaconda3/envs/polymetis-local/lib/python3.8/site-packages/dm_control/mujoco/engine.py", line 407, in from_model
    return cls(data)
  File "/home/panda5/anaconda3/envs/polymetis-local/lib/python3.8/site-packages/dm_control/mujoco/engine.py", line 122, in __init__
    self._reload_from_data(data)
  File "/home/panda5/anaconda3/envs/polymetis-local/lib/python3.8/site-packages/dm_control/mjcf/physics.py", line 530, in _reload_from_data
    super()._reload_from_data(data)
  File "/home/panda5/anaconda3/envs/polymetis-local/lib/python3.8/site-packages/dm_control/mujoco/engine.py", line 370, in _reload_from_data
    self._warnings_before = np.empty_like(self._warnings)
  File "<__array_function__ internals>", line 180, in empty_like
RuntimeError: Caught an unknown exception! 
```

There is a mismatch between the versions installed by the local version of polymetis
and the versions you need
```
pip install mujoco==2.2.1
pip install dm-control==1.0.5
```

## Protobuf version that comes with polymetis gives error
TypeError: bases must be types

to get around it uninstall then install earlier version of protobuf

`` pip uninstall protobuf ``

`` pip install protobuf==3.20.1 `` 

## Cameras are not being detected
If you are using local polymetis to control the robot, make sure you are not running a server.
The server would have taken over the cameras and for security reasons the local env cannot use them.

If no server is running and the cameras are still not detected, restart the NUC

## Issue with Robotiq
If the robotiq server results in this issue: https://github.com/facebookresearch/fairo/issues/1383
Kill the tmux session and restart. Otherwise restart NUC

## User stop mode
If the robot enters user stop mode, the polymetis-server will crash unfortunately. If the robot
needs a manual reset, make sure the training script is halted, then perform the manual reset
and ensure that a new polymetis server is running before continuing.