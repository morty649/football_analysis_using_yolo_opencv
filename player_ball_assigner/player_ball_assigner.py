import sys
import os
sys.path.append('../')
from utils import get_center_of_bbox,measure_distance

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 80

    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player_data in players.items():
            player_bbox = player_data["bbox"]  # Safe accessing
            x1, y1, x2, y2 = player_bbox

            width = x2 - x1

            # Estimate left and right foot positions
            left_foot = (x1 + 0.25 * width, y2)
            right_foot = (x1 + 0.75 * width, y2)

            # Measure distances to ball
            left_distance = measure_distance(ball_position, left_foot)
            right_distance = measure_distance(ball_position, right_foot)
            distance  = max(left_distance,right_distance)

            if distance<self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player





