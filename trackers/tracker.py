from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import os
import sys
import numpy as np
import pandas as pd

sys.path.append('../')

from utils import get_bbox_width,get_center_of_bbox,get_foot_position

class Tracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self,tracks):
        for object,object_tracks in tracks.items():
            for frame_num,track in enumerate(object_tracks):
                for track_id,track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position



    def interpolate_ball_position(self,ball_positions):
        ball_positions = [ x.get(1,{}).get('bbox',[])    for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        #in some frames ball is not tracked for that doing this
        #Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:{'bbox':x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self,frames):
        batch_size = 20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self,frames,read_from_stub=True,stub_path="stub/track_stubs.pkl"):


        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,"rb") as f:
                tracks = pickle.load(f)
            return tracks    
        
        detections = self.detect_frames(frames)
        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }
        for frame_num,detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            #converting to supervision detecting format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #converting goalkeeper to player object
            for object_ind,class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            #Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][0] = {"bbox":bbox}
        if stub_path is not None:
            with open(stub_path,"wb") as f:
                pickle.dump(tracks,f)

            
        return tracks     
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])#we want the circle bottom of player x1,y1 at top left,x2,y2 at bottom right
        x_center,_ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes = (int(width),int(0.35*width)),#ellipse has major ->horizantal and minor ->vertical axes
            color = color,
            angle=0.0,
            startAngle = -45,
            endAngle=235, # we want a 3d effect not full ellipse
            thickness = 2,
            lineType=cv2.LINE_4
        )
        #we also want the player to be identified with a number which is in a rectangle
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15 # + 15 is for random buffer

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect),int(y1_rect)),
                (int(x2_rect),int(y2_rect)),
                color,
                cv2.FILLED
            )
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10  # this is done because to keep number somewhat middle even if track id is big
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, # size
                (0,0,0), # color
                2 # thickness
            )
        return frame
    
    def draw_triangle(self,frame,bbox,color): #for ball
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame,[triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        #Drawing a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay,(1350,850),(1900,970),(255,255,255),-1) # -1 means filled
        alpha = 0.4 # opacity
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        #Get the number of times each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_2_num_frames+team_1_num_frames)

        cv2.putText(frame,f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame
        

    def draw_annotations(self,video_frames,tracks,team_ball_control):
        
        output_video_frames = []
        for frame_num,frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            #Draw players
            for track_id,player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame,player["bbox"],color,track_id)#BGR

                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame,player["bbox"],(0,0,255))
            
            #Draw Referees 
            for _,referee in referee_dict.items():
                frame = self.draw_ellipse(frame,referee["bbox"],(0,255,255))#It is of color yellow

            #Draw Ball
            for track_id,ball in ball_dict.items():
                frame = self.draw_triangle(frame,ball["bbox"],(0,255,0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame,frame_num,team_ball_control)


            output_video_frames.append(frame)
        return output_video_frames
