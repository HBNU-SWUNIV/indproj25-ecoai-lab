# Source Code
## System Architecture & Source Code

본 프로젝트는 ROS 및 Gazebo 환경에서 **TurtleBot3**를 활용하는 자율주행 주차 알고리즘입니다. 

| 모듈 (Module) | 주요 기능 (Description) | 기술 스택 (Tech) |
| :--- | :--- | :--- |
| **Simulation** | Gazebo 기반 주차장 환경 및 TurtleBot3 URDF 모델링 | `ROS1`, `Gazebo` |
| **Perception** | **YOLOv8-lite** 기반 주차선/장애물 인식 및 **Data Augmentation** (야간/우천 대응) | `PyTorch`, `OpenCV` |
| **Interface** | 한국어 음성 명령 인식 및 주차 모드(평행/수직) 제어 | `SpeechRecognition` |
| **Planning** | Hough Transform 기반 라인 추출 및 주차 경로(Path) 생성 알고리즘 | `Python`, `NumPy` |



| 아키텍처 |
| :---: | 
| <img src="https://github.com/user-attachments/assets/0199bfd5-bc20-44b9-b3fc-65a73eb404fa" width="100%"> |
