from robot_env import RobotEnv
import numpy as np
from controllers import XboxController

class Workspace(object):
    def __init__(self):

        # initialize robot environment
        self.env = RobotEnv(hz=10,
                            ip_address='172.16.0.10',
                            randomize_ee_on_reset=False,
                            pause_after_reset=False,
                            hand_centric_view=True,
                            third_person_view=True,
                            qpos=True,
                            ee_pos=True,
                            local_cameras=False)
        self.controller = XboxController(DoF=3)

    def momentum(self, delta, prev_delta):
        """Modifies action delta so that there is momentum (and thus less jerky movements)."""
        prev_delta = np.asarray(prev_delta)
        gamma = 0.15 # higher => more weight for past actions
        return (1 - gamma) * delta + gamma * prev_delta

    def run(self):
        self.env.reset()
        prev_action = np.zeros(4)
        print('free controlling now')
        while True:
            # smoothen the action
            xbox_action = self.controller.get_action()
            smoothed_pos_delta = self.momentum(xbox_action[:3], prev_action[:3])
            action = np.append(smoothed_pos_delta, xbox_action[3]) # concatenate with gripper command
            obs, _, _, _ = self.env.step(action)
            print(obs['lowdim_ee'])
            prev_action = action

if __name__ == '__main__':
    workspace = Workspace()
    workspace.run()