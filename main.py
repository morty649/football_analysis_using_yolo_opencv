from utils import read_video,save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
def main():
    #Read video
    video_frames = read_video('input_videos/video-15-seconds.mp4')# 10 seconds

    #video_frames = read_video('input_videos/08fd33_4.mp4')

    #initialize tracker
    tracker = Tracker('models/best(2).pt')  #best(2) is large and best is nano

    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="stub/track_stubs.pkl")

    #get object positions
    tracker.add_position_to_tracks(tracks)

    #camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,read_from_stub=True,stub_path='stubs/camera_movement_stub.pkl')


    camera_movement_estimator.add_adjust_position_to_tracks(tracks,camera_movement_per_frame)

    #view transformation
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    #Interpolate ball missing positions 
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    #speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks['players'][0]) #only first frame because memory management
    #num_frames = min(len(video_frames), len(tracks['players']))
    num_frames = len(tracks['players'])

    for frame_num in range(num_frames):
        if frame_num >= len(video_frames):
            break
        player_track = tracks['players'][frame_num]
        for player_id, player_data in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], player_data['bbox'], player_id)
            player_data['team'] = team
            player_data['team_color'] = team_assigner.team_colors[team]


    # for frame_num in range(num_frames):
    #     player_track = tracks['players'][frame_num]
    #     for player_id, player_data in player_track.items():
    #         team = team_assigner.get_player_team(video_frames[frame_num], player_data['bbox'], player_id)
    #         player_data['team'] = team
    #         player_data['team_color'] = team_assigner.team_colors[team]

    #Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num,player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)

        # if assigned_player != -1:
        #     tracks['players'][frame_num][assigned_player]['has_ball'] = True
        #     team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        # else:
        #     team_ball_control.append(team_ball_control[-1])
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team = tracks['players'][frame_num][assigned_player].get('team', -1)
            team_ball_control.append(team)
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)

    team_ball_control = np.array(team_ball_control)





    #drawing ellipses around 
    #drawing all objects
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

    #draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    #draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)


    #save video
    save_video(output_video_frames,'output_videos/output-15-seconds.avi')
    print("Video saved to output_videos/output_video.avi")


if __name__ == '__main__':
    main()