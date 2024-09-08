import random
import sys
import traceback
from collections import deque
from enum import Enum
from os import path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

RENDERING = False
RESET_THRESHOLD = 10
SPRINT_STATE_LOCK_P0 = 25
SPRINT_STATE_LOCK_P1 = 45
VICINITY_RADIUS = 10

WORLD_SCALING_FACTOR = 1
WORLD_X_CORRECTION = 0
WORLD_Y_CORRECTION = 0
ATTACK_CONE_ANGLE = 35  # +/- 35 degrees from kart2goal vector

SCREEN_X = 400
SCREEN_Y = 300

SCREEN_CENTER = np.float32([200, 150])
STADIUM_CENTER = np.float32([0, 0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def norm(vector):
    return np.linalg.norm(vector)


def dist(pos1, pos2):
    return norm(pos2 - pos1)


def to_numpy(pos):
    return np.float32([pos[0], pos[2]]) if len(pos) == 3 else np.float32([pos[0], pos[1]])


def to_vector(np1, np2):
    return np2 - np1


def to_image(x, proj, view):
    p = proj @ view @ np.array(list(x) + [1])
    return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)


def to_world(s, proj, view, dist):
    screen = np.array(list(s) + [-1, 1])
    camera = np.linalg.inv(proj) @ screen
    camera = camera[0:3]/camera[3]
    camera = (camera/np.linalg.norm(camera))*dist
    camera = np.array(list(camera)+[1])
    world = np.linalg.inv(view) @ camera
    world = world[0:3]/world[3]
    return world


def angle_diff(vector1, vector2):
    return np.rad2deg(np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0]))


class State(Enum):
    START = 0
    VISIBLE = 1
    CONTACT = 2
    LOST = 3
    STUCK = 4
    GOAL = 5


class Role(Enum):
    OFFENSE = 0
    DEFENSE = 1


class DriveParams:
    DRIFT_ANGLE = 45
    DRIFT_RATIO = 0.2
    MAX_SEARCH_VEL = 8
    MAX_SEARCH_ACC = 8
    MAX_VEL = 20
    MAX_ACC = 10
    MAX_ATTACK_VEL = 15
    MAX_ATTACK_ACC = 5
    OVERSTEER = 4
    UNDERSTEER = 2


class Player:

    # ================================================= Initialize =================================================== #
    last_known_puck_pos = SCREEN_CENTER

    @classmethod
    def update_puck_pos(cls, pos):
        cls.last_known_puck_pos = pos

    def __init__(self, team, player, render):
        # select kart
        self.kart = "wilber" if player == 0 else "nolok"
        # set up parameters

        self.player_id = player
        self.render = render
        self.past_kart_poss = deque(maxlen=5)
        self.past_puck_screens = deque(maxlen=5)
        self.past_state = deque(maxlen=5)
        self.past_actions = deque(maxlen=5)

        # State lock stuff
        self.state = State.START
        self.state_lock = True
        self.action_name = "SPRINTING"
        self.locked_action = self.sprint_action
        self.sprint_state_lock = SPRINT_STATE_LOCK_P0 if player == 0 else SPRINT_STATE_LOCK_P1
        self.state_lock_turns = self.sprint_state_lock
        self.puck_lost_count = 0
        self.puck_visible_count = 0

        # Velocity values
        self.current_vel = 0
        self.target_vel = 25
        self.last_seen_puck_screen = SCREEN_CENTER
        self.update_puck_pos(STADIUM_CENTER)
        self.kart_screen = SCREEN_CENTER

        # Camera projection variables
        self.kart_view = None
        self.kart_proj = None
        self.kart_world = None
        self.puck_pos = None

        # Select Team
        self.team = team
        # Determine ours and their goals
        if self.team == 0:
            self.our_goal_left = to_numpy((-10, -64))
            self.our_goal_center = to_numpy((0, -64))
            self.our_goal_right = to_numpy((10, -64))
            self.their_goal_left = to_numpy((-10, 64))
            self.their_goal_center = to_numpy((0, 64))
            self.their_goal_right = to_numpy((10, 64))
        else:
            self.our_goal_left = to_numpy((-10, 64))
            self.our_goal_center = to_numpy((0, 64))
            self.our_goal_right = to_numpy((10, 64))
            self.their_goal_left = to_numpy((-10, -64))
            self.their_goal_center = to_numpy((0, -64))
            self.their_goal_right = to_numpy((10, -64))

        # assign offense and defense
        if self.player_id == 0:
            self.role = Role.DEFENSE
        else:
            self.role = Role.OFFENSE

    # ==================================================== Checks ==================================================== #

    def check_kart_reset(self):
        if len(self.past_kart_poss) <= 0:
            return True
        last_loc = self.past_kart_poss[-1]
        x_diff = abs(last_loc[0] - self.kart_pos[0])
        y_diff = abs(last_loc[1] - self.kart_pos[1])
        if x_diff > RESET_THRESHOLD or y_diff > RESET_THRESHOLD:
            print(f"Player {self.player_id} reset!")
            self.state_lock_turns = 0
            self.state_lock = False
            return True
        return False

    def check_puck_visible(self):
        # if in the goal puck won't be visible
        if self.check_kart_in_goal():
            # In the Blue goal facing away from stadium center
            # print("-----------------in Goal")
            if self.kart_pos[1] > 0 and self.kart_front[1] - self.kart_pos[1] < -0.3:
                # print("----------Not visible in Blue Goal")
                return False
            # In the Red goal facing away from stadium center
            elif self.kart_pos[1] < 0 and self.kart_front[1] - self.kart_pos[1] > 0.3:
                # print("----------Not visible in Red Goal")
                return False

        if sum(x is not None for x in self.past_puck_screens) > 3:
            self.puck_lost_count -= 1
            self.puck_visible_count += 1
        if self.puck_visible_count > self.puck_lost_count:
            self.puck_lost_count = 0
            self.puck_visible_count = 0
            return True
        return False

    def check_puck_lost(self):
        if sum(x is None for x in self.past_puck_screens) > 4:
            self.puck_lost_count += 1
            self.puck_visible_count -= 1

        if self.puck_lost_count > 10:
            self.update_puck_pos(STADIUM_CENTER)

        if self.puck_lost_count > self.puck_visible_count:
            self.puck_lost_count = 0
            self.puck_visible_count = 0
            return True
        return False

    def check_kart_near_puck(self):
        if (
            self.puck_screen is None
            or self.kart_screen is None
            or self.last_seen_puck_screen is None
            or len(self.past_kart_poss) < 5
        ):
            return False

        distance = dist(self.puck_screen, self.kart_screen)
        prev_distance = dist(self.last_seen_puck_screen, self.kart_screen)
        if sum(x is None for x in self.past_kart_poss) < 1 and prev_distance < 60 and distance < 55:
            return True
        return False

    def check_kart_in_goal(self):
        return True if abs(self.kart_pos[0]) < 11 and (self.kart_pos[1] > 63.8 or self.kart_pos[1] < -63.8) else False

    def check_state(self):
        if self.check_kart_reset():
            # print(f"Player {self.player_id} : Position Reset")
            # input("Ack This..")
            return self.sprint_action

        elif self.check_kart_in_goal():
            # print(f"Player {self.player_id} : Kart in goal")
            return self.escape_goal_action

        elif self.check_puck_visible():
            if self.check_kart_near_puck():
                # print(f"Player {self.player_id} : Kart near puck")
                return self.attack_action
            # print(f"Player {self.player_id} : Puck visible")
            return self.chase_action

        elif self.check_puck_lost():
            # print(f"Player {self.player_id} : Puck lost")
            return self.search_action

        return self.search_action

    # ==================================================== Actions =================================================== #

    def sprint_action(self, action):
        self.action_name = "SPRINTING"
        if not self.state_lock:
            self.state_lock = True
            self.state_lock_turns = self.sprint_state_lock
            self.locked_action = self.sprint_action
        if self.state_lock_turns < 1:
            action["fire"] = True

        action["acceleration"] = 1
        action["brake"] = False
        action["steer"] = 0
        action["nitro"] = True
        return action

    def chase_action(self, action):
        self.action_name = "CHASING"
        if self.puck_screen is None:
            self.state_lock = False
            self.state_lock_turns = 0
            return action

        if not self.state_lock:
            self.state_lock = True
            self.state_lock_turns = 30
            self.locked_action = self.chase_action

        # Pretty bad approximation but let's see.
        distance = dist(self.puck_screen, self.kart_screen)
        acceleration = (DriveParams.MAX_VEL**2 - self.current_vel**2) / (2 * distance)

        kart_vector = to_vector(SCREEN_CENTER, self.kart_screen)
        puck_vector = to_vector(self.puck_screen, self.kart_screen)

        steer_angle = angle_diff(kart_vector, puck_vector)
        # print(f"Angle_diff : {steer_angle}, Dist : {distance}")

        steer_val = DriveParams.OVERSTEER * steer_angle / 90.0
        acc_val = acceleration / DriveParams.MAX_ACC if acceleration < DriveParams.MAX_ACC else 1.0

        # Check if braking is needed
        brake_val = False
        if self.current_vel > DriveParams.MAX_VEL:
            brake_val = True
            acc_val = 0.0

        # Check if drifting is needed
        drift_val = False
        drift_vel = DriveParams.DRIFT_RATIO * DriveParams.MAX_VEL
        if np.abs(steer_angle) > DriveParams.DRIFT_ANGLE and self.current_vel > drift_vel:
            brake_val = True
            drift_val = True

        # print(f"Chasing Puck at {self.puck_screen}, Kart {self.kart_screen}, velocity {self.current_vel:.2f}")
        action["steer"] = steer_val
        action["acceleration"] = acc_val
        action["brake"] = brake_val
        action["drift"] = drift_val
        return action

    def attack_action(self, action):
        self.action_name = "ATTACKING"
        if self.puck_screen is None or self.puck_pos is None:
            self.state_lock = False
            self.state_lock_turns = 0
            return action

        if not self.state_lock:
            self.state_lock = True
            self.state_lock_turns = 5
            self.locked_action = self.attack_action

        # print(f"Player {self.player_id} : Predicted Puck Pos : {self.puck_pos}")
        # print(f"Player {self.player_id} : Actual    Kart Pos : {self.kart_pos}")
        distance = dist(self.puck_pos, self.kart_pos)
        acceleration = (DriveParams.MAX_VEL**2 - self.current_vel**2) / (4 * distance)

        kart2puck_vector = to_vector(self.puck_pos, self.kart_pos)
        kart2goal_vector = to_vector(self.their_goal_center, self.kart_pos)
        attack_angle = angle_diff(kart2goal_vector, kart2puck_vector)

        goal2puck_vector = to_vector(self.puck_pos, self.their_goal_center)
        goal2kart_vector = to_vector(self.kart_pos, self.their_goal_center)
        goal2puck_angle = angle_diff(goal2puck_vector, goal2kart_vector)

        if abs(attack_angle) > ATTACK_CONE_ANGLE:
            # print("Give up for now !!!")
            return action
        else:
            steer_angle = -1*(attack_angle + goal2puck_angle)
            steer_val = steer_angle / (90.0*DriveParams.UNDERSTEER)

        # print(f"Attack Angle  : {attack_angle}, Goal2Puck_Angle : {goal2puck_angle} Dist : {distance} Str: {steer_val}")
        acc_val = acceleration / DriveParams.MAX_ATTACK_ACC if acceleration < DriveParams.MAX_ATTACK_ACC else 0.5

        # Check if braking is needed
        brake_val = False
        if self.current_vel > DriveParams.MAX_ATTACK_VEL:
            brake_val = True
            acc_val = 0.0

        # Check if drifting is needed
        drift_val = False
        drift_vel = DriveParams.DRIFT_RATIO * DriveParams.MAX_ATTACK_VEL
        if np.abs(steer_angle) > DriveParams.DRIFT_ANGLE and self.current_vel > drift_vel:
            brake_val = True
            drift_val = True

        # print(f"Chasing GOAL at {self.puck_screen}, Kart {self.kart_screen}, velocity {self.current_vel:.2f}")
        action["steer"] = steer_val
        action["acceleration"] = acc_val
        action["brake"] = brake_val
        action["drift"] = drift_val
        return action

    def search_action(self, action):
        self.action_name = "SEARCHING"
        if not self.state_lock:
            self.state_lock = True
            self.state_lock_turns = 2
            self.locked_action = self.search_action

        distance = dist(self.last_known_puck_pos, self.kart_pos)

        dx = self.last_known_puck_pos[0] - STADIUM_CENTER[0]
        dy = self.last_known_puck_pos[1] - STADIUM_CENTER[1]

        da = self.kart_front[0] - STADIUM_CENTER[0]
        db = self.kart_front[1] - STADIUM_CENTER[1]

        target_angle = np.rad2deg(np.arctan2(dy, dx))
        current_angle = np.rad2deg(np.arctan2(db, da))
        steer_angle = target_angle - current_angle
        steer_angle = -1*np.rad2deg((np.deg2rad(steer_angle) + np.pi) % (2 * np.pi) - np.pi)

        # print(f"Puck Last : {self.last_known_puck_pos}, {target_angle=}, {current_angle=}, {steer_angle=}")

        steer_val = DriveParams.OVERSTEER * steer_angle/90
        acceleration = (DriveParams.MAX_VEL**2 - self.current_vel**2) / (2 * distance)
        acc_val = acceleration / DriveParams.MAX_SEARCH_ACC if acceleration < DriveParams.MAX_SEARCH_ACC else 1.0

        # Check if braking is needed
        brake_val = False
        if self.current_vel > DriveParams.MAX_SEARCH_VEL:
            brake_val = True
            acc_val = 0.0

        # Check if drifting is needed
        drift_val = False
        drift_vel = DriveParams.DRIFT_RATIO * DriveParams.MAX_SEARCH_VEL
        if np.abs(steer_angle) > DriveParams.DRIFT_ANGLE and self.current_vel > drift_vel:
            brake_val = True
            drift_val = True

        action["steer"] = steer_val
        action["acceleration"] = acc_val
        action["brake"] = brake_val
        action["drift"] = drift_val
        return action

    def escape_goal_action(self, action):
        self.action_name = "ESCAPING"
        if not self.state_lock:
            self.state_lock = True
            self.state_lock_turns = 20
            self.locked_action = self.escape_goal_action

        if self.last_seen_puck_screen is None:
            self.last_seen_puck_screen = np.float32([0, 0])

        # In the Blue goal
        if self.kart_pos[1] > 0:
            # If facing backwards, go backwards
            if self.kart_front[1] - self.kart_pos[1] > -0.3:
                action["acceleration"] = 0
                action["brake"] = True
                action["steer"] = 0.7 if self.last_seen_puck_screen[0] < self.kart_pos[0] else -0.7
            # Otherwise you're facing forwards, so accelerate
            else:
                action["acceleration"] = 1
                action["brake"] = False
                action["steer"] = -1 if self.last_seen_puck_screen[0] > self.kart_pos[0] else 1

        # In the Red goal
        else:
            # If facing backwards, go backwards
            if self.kart_front[1] - self.kart_pos[1] < 0.3:
                action["acceleration"] = 0
                action["brake"] = True
                action["steer"] = -0.7 if self.last_seen_puck_screen[0] < self.kart_pos[0] else 0.7

            # Otherwise you're facing forwards, so accelerate
            else:
                action["acceleration"] = 1
                action["brake"] = False
                action["steer"] = 1 if self.last_seen_puck_screen[0] < self.kart_pos[0] else -1

        if abs(self.kart_pos[1]) > 69:
            action["steer"] = action["steer"] * ((10 - abs(self.kart_pos[0])) / 10)
        action["nitro"] = False
        return action

    def stuck_action(self, action):
        self.action_name = "DISLODGING"
        if not self.state_lock:
            self.state_lock = True
            self.state_lock_turns = 10
            self.locked_action = self.stuck_action
        action["acceleration"] = 0
        action["brake"] = True
        action["steer"] = 0
        return action

    def random_action(self, action):
        self.action_name = "YOLOING"
        if not self.state_lock:
            self.state_lock = True
            self.state_lock_turns = 5
            self.locked_action = self.random_action
            self.random_action_reverse = True if random.random() > 0.4 else False
        if self.random_action_reverse:
            action["acceleration"] = 0
            action["brake"] = True
        else:
            action["acceleration"] = 1
            action["brake"] = False
        action["steer"] = 2*random.random()-1
        return action

    def stop_action(self, action):
        self.action_name = "STOPPING"
        if not self.state_lock:
            self.state_lock = True
            self.state_lock_turns = 10
            self.locked_action = self.stop_action

        if len(self.past_kart_poss) < 1:
            return action
        kart_pos_to_kart_front = to_vector(self.kart_front, self.kart_pos)
        past_pos_to_kart_front = to_vector(self.kart_front, self.past_kart_poss[-1])
        np.dot(past_pos_to_kart_front, kart_pos_to_kart_front)
        # print(direction)

        if self.current_vel > 0:
            action["brake"] = True
            action["acceleration"] = 0.001

        action["steer"] = 0
        action["nitro"] = False
        return action

    # ================================================= State Info =================================================== #

    def get_state_info(self, puck_screen, player_state):
        self.puck_screen = to_numpy(puck_screen) if puck_screen is not None else None
        self.kart_world = np.float32(
            [
                player_state["kart"]["location"][0],
                player_state["kart"]["location"][1],
                player_state["kart"]["location"][2],
            ]
        )
        self.kart_proj = np.array(player_state["camera"]["projection"]).T
        self.kart_view = np.array(player_state["camera"]["view"]).T
        kart_img_coord = to_image(self.kart_world, self.kart_proj, self.kart_view)
        self.kart_screen = ((SCREEN_X / 2) * (1 + kart_img_coord[0]), (SCREEN_Y / 2) * (1 + kart_img_coord[1]))
        self.kart_pos = to_numpy(player_state["kart"]["location"])
        self.kart_front = to_numpy(player_state["kart"]["front"])
        self.current_vel = np.linalg.norm(player_state["kart"]["velocity"])

        self.last_seen_puck_screen = (
            self.puck_screen
            if self.puck_screen is not None and sum(puck_screen is None for puck_screen in self.past_puck_screens) < 1
            else None
        )
        self.past_puck_screens.append(self.puck_screen)
        self.past_kart_poss.append(self.kart_pos)
        self.puck_pos = None

        if self.puck_screen is not None and self.check_kart_near_puck():
            world_dist = dist(SCREEN_CENTER, self.puck_screen) * WORLD_SCALING_FACTOR
            puck_from_screen_center = self.kart_screen - np.array([200, 150])
            self.puck_pos = to_world(puck_from_screen_center, self.kart_proj, self.kart_view, world_dist)
            self.puck_pos = np.array((self.puck_pos[0]+WORLD_X_CORRECTION, self.puck_pos[2]+WORLD_Y_CORRECTION))
            # self.last_known_puck_pos = self.puck_pos
            self.update_puck_pos(self.puck_pos)
            # print(f"Player {self.player_id} : Predicted Puck Pos : {self.puck_pos}")
            # print(f"Player {self.player_id} : Actual Kart Pos    : {self.kart_pos[0]:.2f}, {self.kart_pos[1]:.2f}")

    # ================================================= Controller =================================================== #
    def act(self, puck_screen, player_state, image=None):
        action = {
            "acceleration": 1,
            "brake": False,
            "drift": False,
            "nitro": False,
            "rescue": False,
            "fire": False,
            "steer": 0,
        }

        self.get_state_info(puck_screen, player_state)
        suggested_action = self.check_state()
        # check can override by setting state_lock_turns == 0 to force controller to pick suggested_action.
        if self.state_lock_turns == 0:
            self.state_lock = False
            next_action = suggested_action
        # otherwise by default play the last action until the lock is over.
        else:
            self.state_lock_turns -= 1
            next_action = self.locked_action

        action = next_action(action)
        self.action_name += f" | LOCATION: {self.kart_pos[0]:.2f}, {self.kart_pos[1]:.2f} "
        (
            self.render.update(self.player_id, image, action, puck_screen, self.kart_screen, self.action_name)
            if self.render
            else None
        )
        return action


# ------------------------------------------------- Helper Utils ----------------------------------------------------- #

class Render:
    def __init__(self) -> None:
        plt.ion()
        self.fig = [None, None]
        self.axs = [None, None]
        self.img = [None, None]
        self.puck = [None, None]
        self.kart = [None, None]
        self.fig[0], self.axs[0] = plt.subplots()
        self.fig[1], self.axs[1] = plt.subplots()
        # self.blank_img = Image.open("image_agent/start.png")
        self.blank_img = Image.fromarray(np.zeros((SCREEN_Y, SCREEN_X)))
        self.img[0] = self.axs[0].imshow(self.blank_img)
        self.img[1] = self.axs[1].imshow(self.blank_img)
        self.puck[0] = patches.Circle((200, 150), 3, linewidth=3, edgecolor="none", facecolor="none")
        self.puck[1] = patches.Circle((200, 150), 3, linewidth=3, edgecolor="none", facecolor="none")
        self.kart[0] = patches.Circle((200, 150), 3, linewidth=3, edgecolor="none", facecolor="none")
        self.kart[1] = patches.Circle((200, 150), 3, linewidth=3, edgecolor="none", facecolor="none")
        self.axs[0].add_patch(self.puck[0])
        self.axs[0].add_patch(self.kart[0])
        self.axs[1].add_patch(self.puck[1])
        self.axs[1].add_patch(self.kart[1])

    def update(self, player, image, action, puck_screen, kart_screen, action_name):
        self.kart[player].set(center=kart_screen, edgecolor="g", facecolor="y")
        if puck_screen is not None:
            self.puck[player].set(center=puck_screen, edgecolor="y")
        else:
            self.puck[player].set(edgecolor="none")
        action_str = f"Player {player} : "
        action_str += f" [ Acc: {action['acceleration']:.2f} | Str: {action['steer']:.2f} | {action_name}]"
        action_str += f"{' @--REVERSE--@ ' if action['brake'] and action['acceleration'] == 0 else ''}"
        action_str += f"{' !!! BRAKE !!! ' if action['brake'] and action['acceleration'] != 0 else ''}"
        action_str += f"{' <<< DRIFT >>> ' if action['drift'] else ''}"
        action_str += f"{' *** NITRO *** ' if action['nitro'] else ''}"
        action_str += f"{' <!> FIRE  <!> ' if action['fire'] else ''}"

        self.axs[player].set_title(action_str)
        self.img[player].set_data(image)
        self.fig[player].canvas.flush_events()


class Detect:
    def __init__(self):
        self.max_pool_ks = 2
        self.min_score = -5
        self.max_det = 1

    def extract_peak(self, heatmap, max_pool_ks, min_score, max_det):
        """
        Your code here.
        Extract local maxima (peaks) in a 2d heatmap.
        @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
        @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
        @min_score: Only return peaks greater than min_score
        @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                 heatmap value at the peak. Return no more than max_det peaks per image
        """
        best, indices = torch.nn.functional.max_pool2d(
            input=heatmap[None, None],
            kernel_size=max_pool_ks,
            padding=max_pool_ks // 2,
            stride=1,
            return_indices=True,
        )

        # Cropping this 301, 401 tensor....will worry about it later.
        best = best[:, :, :300, :400]
        indices = indices[:, :, :300, :400]

        peak = torch.ge(heatmap, best)
        valid = torch.gt(best, min_score)
        cutoff = torch.eq(peak.float(), 1.0)
        mask = torch.logical_and(valid, cutoff)

        indices, best = indices[mask], best[mask]
        peaks, i = torch.topk(best, min(max_det, len(best)))

        peak_list = []
        for peak, index in zip(peaks, indices[i]):
            cx = index % heatmap.shape[1]
            cy = index // heatmap.shape[1]
            peak_list.append((peak.item(), cx.item(), cy.item()))

        return peak_list

    def __call__(self, heatmap, max_pool_ks=None, min_score=None, max_det=None):
        max_pool_ks = self.max_pool_ks if max_pool_ks is None else max_pool_ks
        min_score = self.min_score if min_score is None else min_score
        max_det = self.max_det if max_det is None else max_det
        peak_list = []
        for peak in self.extract_peak(heatmap[0], max_pool_ks, min_score, max_det):
            peak_list.append((*peak, 0, 0))
        return peak_list


class ToTensor(object):
    def __call__(self, image, *args):
        return (F.to_tensor(image),) + args


# ------------------------------------------------- Team Class ------------------------------------------------------- #

class Team:
    agent_type = "image"

    def __init__(self):
        self.kart = "sara_the_racer"
        self.model = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), 'player.pt'))
        self.model.eval().to(device)
        self.players = []
        self.render = Render() if RENDERING else None
        self.transform = ToTensor()
        self.detector = Detect()

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        for i in range(num_players):
            self.players.append(Player(team, i, self.render))
        return [p.kart for p in self.players]

    def act(self, player_state, player_image):
        try:
            actions = []
            for player in range(self.num_players):
                # predict puck position
                image = Image.fromarray(player_image[player])
                img_tensor = self.transform(image)[0].to(device)
                heatmap = self.model(img_tensor[None]).squeeze(0)
                pred = self.detector(heatmap)
                puck_found = len(pred) > 0 and len(pred[0]) > 0
                puck_screen = (pred[0][1], pred[0][2]) if puck_found else None
                action = self.players[player].act(puck_screen, player_state[player], image)
                actions.append(action)
            return actions
        except Exception as e:                                        # This is truly the world's most reliable Team !!!
            self._error = 'Failed to act: {}'.format(str(e))
            traceback.print_exception(*sys.exc_info())
            return [dict(acceleration=0, steer=0)] * self.num_players

# -------------------------------------------------    END     ------------------------------------------------------- #