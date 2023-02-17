# iris_robots: Franka
## Adaptation of polymetis for Franka 

## Setup instructions
For all experiments we utilize a Franka Emika Panda research robot: https://www.franka.de/research. The only modification we made in terms of hardware
is using a Robotiq 2F-85 gripper: https://robotiq.com/products/2f85-140-adaptive-robot-gripper, as we found the gripper with the Franka was prone
to failure when running longer RL experiments (1 hour+)

We use the polymetis environment wrapper to work with our Franka robot, with the instructions to download found here: https://facebookresearch.github.io/fairo/polymetis/. We thank the authors for nice codebase / wrapper to work with!


Our workflow assumes a controller computer. For our use case we used an intel NUC machine as recommended by the polymetis instructions. If working on a workstation / separate machine from controller (NUC), make a conda env using `conda create -n polymetis python=3.8` workflow.

We recommend the local installation path option for polymetis has using conda forge caused us compilation issues.

Need to also install dm_control (latest verison) and dm_robotics package to
use the IK solver

``pip install dm-robotics-moma``

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

# Demo Collection
For demo collection, we utilize an Xbox One controller which is directly plugged into the NUC controller computer.
You can find one [here](https://support.xbox.com/en-US/help/hardware-network/controller/xbox-one-wireless-controller). If you would like to get a sense of the commands, try running 
```
python free_xbox_control.py
```
You can also modify the robot_env to use either 4-DoF or 5-DoF control, with an optional degree of freedom in
rotating the orientation of the gripper up to 180 degrees. The commands are
```
'A': Close gripper. Must be held to continue closing gripper, and once released the gripper will open
'X': End demo collection episode early.
'Left stick (right to left)' : Move end-effector along y-axis with a delta depending on how far the stick is moved
'Left stick (upwards to downwards)' : Move end-effector along x-axis with a delta depending on how far the stick is moved
'Right trigger': Move end-effector upwards along z-axis with a delta depending on how much the trigger is pressed
'Left trigger': Move end-effector downwards along z-axis with a delta depending on how much the trigger is pressed
'Right stick (right to left)': Rotate end-effector clockwise/counter-clockwise depending on which direction the stick is moved
only used for 5 DoF control
```

The demo collection script used for the main experiments can be run by
```
python demo_collect_fb.py demos
```

Which will create a folder named 'demos' where all demo data and GIFs of data collection are stored.
If a folder with the same name exists before, the script first prompts the user with the following text:
```
A demo file already exists. Continue appending to it? (y) or (n): 
```

Be careful that you do NOT enter n on your keyboard if you wish to avoid overwriting your demos!

Afterwards demo collection will begin and you can control the robot with the Xbox controller for
a limited number of timesteps, which we configured per task but which can be changed in the ```robot_env.py```
file. If you wish to end the demo early, or you reach the end of the episode, data collection will stop and
you will recieve the following prompt:

``` Save current episode?: (f) for forward buffer, (b) for backward buffer and (d) to discard episode and (q) to save and quit: ```
Enter either `f` or `b` depending on whether the demo you collected is for the forward / backward buffer as stated above. If you 
would like another prompt which waits for you to start the next episode with a keyboard entry enter `fj` or `bj`. If you enter
`d` or `dj` you will discard the demo either directly continuing into the start of collecting the next demo or pausing before continuing.

If you do not enter either `f` or `b` then no demo will be saved. If you would like to save and quit you must enter `fq` or `bq`.

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

# Error FAQ
## Mujoco Engine Error
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
The server would have taken over the cameras and for security reasons the OS will not let the local env cannot use them.

If no server is running and the cameras are still not detected, restart the NUC

## Issue with Robotiq
If the robotiq server results in this issue: https://github.com/facebookresearch/fairo/issues/1383
Kill the tmux session and restart. Otherwise restart NUC

## User stop mode
If the robot enters user stop mode, the polymetis-server will crash unfortunately. If the robot
needs a manual reset, make sure the training script is halted, then perform the manual reset
and ensure that a new polymetis server is running before continuing.