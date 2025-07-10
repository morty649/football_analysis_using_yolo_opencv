Football Analysis Using YOLOv8 & OpenCV
This project is a complete computer vision pipeline for analyzing football match footage using YOLOv8, OpenCV, and Optical Flow. It detects players, assigns teams, tracks the ball, estimates player speed and total distance covered, and compensates for camera motion to generate fully annotated match insights.

Features
Player Detection using YOLOv8 (custom-trained model)

Team Assignment via jersey color clustering

Speed Estimation for each player using position deltas over time

Total Distance Covered by each player

Ball Possession Tracking (identifying who controls the ball per frame)

Camera Motion Compensation using Optical Flow and goodFeaturesToTrack

Annotated output video with:

Player IDs

Speed values

Team indicators

Ball control status

Modular pipeline structure

Tech Stack
YOLOv8 (via ultralytics Python package)

OpenCV for tracking, feature detection, and video processing

NumPy for numerical calculations

Python modules for:

Inference (yolo_inference.py)

Camera motion estimation

Player-ball assignment

Speed and distance computation

Project Structure
football_analysis_using_yolo_opencv/
├── main.py # Main pipeline script
├── yolo_inference.py # YOLO model wrapper
├── speed_and_distance_estimator/ # Speed and distance logic
├── player_ball_assigner/ # Player-ball linking module
├── camera_movement_estimator/ # Optical flow for camera stabilization
├── input_videos/ # Input football videos
├── output_videos/ # Output annotated results
├── models/ # YOLO weights (e.g., best.pt)
├── stub/ # Preprocessed or intermediate data
├── development_and_analysis/ # Jupyter notebook experiments
└── README.md

Place your input video in the input_videos/ directory.

Run the main script:
python main.py --video input_videos/your_video.mp4

The output will be saved in output_videos/.

Output Description
The final output is a fully annotated video that includes:

Detected players with IDs

Ball possession tracking

Player speed and distance estimations

Tracking stabilized against camera movements

What I Learned
Real-world application of YOLOv8 and OpenCV

Motion stabilization using Optical Flow

Handling complex camera movements in sports footage

Designing modular, scalable pipelines for video analysis

References
Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics

OpenCV Documentation: https://docs.opencv.org/

https://youtu.be/neBZ6huolkg?si=YTIYQdzDEnZTzeCG

Feedback & Contributions
Feedback, improvements, or collaborations are welcome. Please feel free to open an issue or reach out.

Connect
LinkedIn: https://www.linkedin.com/in/maruthi-enugula

GitHub Repository: https://github.com/morty649/football_analysis_using_yolo_opencv

