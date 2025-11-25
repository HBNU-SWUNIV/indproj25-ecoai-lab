#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomous Parking Robot – Operating Software Skeleton
======================================================

자율주행 주차로봇 운영 소프트웨어 성능 테스트 및 최적화
- YOLOv8 기반 객체 검출
- 주차선 인식 (IPM + Hough Transform)
- 3단계 주차 가능 판정 (가능/보류/불가)
- ROS / Gazebo 연동 뼈대
- UI 및 로그 자동화

실제 프로젝트에서 사용할 때는
1) 경로, 토픽 이름, 카메라 파라미터
2) YOLO 모델 경로
3) Gazebo/실로봇 환경
에 맞게 수정해서 사용하면 됩니다.
"""

import os
import cv2
import sys
import time
import math
import json
import queue
import threading
import datetime
import logging
from enum import IntEnum, Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

try:
    # 선택사항: ultralytics 패키지 설치되어 있을 때만 사용
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import rospy
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Twist
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except Exception:
    ROS_AVAILABLE = False


# ---------------------------------------------------------------------------
# 설정 및 상수 정의
# ---------------------------------------------------------------------------

class EnvironmentProfile(str, Enum):
    DAY = "day"
    NIGHT = "night"
    RAIN = "rain"


class ParkingDecision(IntEnum):
    POSSIBLE = 0      # GREEN
    HOLD = 1          # YELLOW
    IMPOSSIBLE = 2    # RED


@dataclass
class CameraConfig:
    width: int = 1280
    height: int = 720
    fov_deg: float = 90.0
    # IPM (Top-view)용 변환점: 실제 카메라마다 맞게 조정 필요
    src_points: np.ndarray = field(default_factory=lambda: np.float32([
        [250, 450], [1030, 450], [50, 720], [1230, 720]
    ]))
    dst_points: np.ndarray = field(default_factory=lambda: np.float32([
        [0, 0], [640, 0], [0, 480], [640, 480]
    ]))


@dataclass
class HoughConfig:
    rho: float = 1.0
    theta: float = np.pi / 180
    threshold: int = 40
    min_line_length: int = 40
    max_line_gap: int = 20


@dataclass
class ParkingThresholds:
    min_width_m: float = 2.1     # 최소 주차 폭
    max_width_m: float = 3.0     # 최대 주차 폭
    max_parallel_deg: float = 8  # 평행 허용 각도
    risk_hold: float = 0.4       # HOLD 기준 리스크 스코어
    risk_impossible: float = 0.7 # IMPOSSIBLE 기준 리스크 스코어
    obstacle_margin_m: float = 0.4  # 장애물과의 최소 여유거리


@dataclass
class YOLOConfig:
    model_path: str = "./weights/yolov8n.pt"  # 실제 학습 모델 경로로 교체
    conf_thres: float = 0.4
    iou_thres: float = 0.5
    device: str = "cuda"  # 또는 "cpu"
    classes_vehicle: Tuple[int, ...] = (2, 3, 5, 7)  # COCO: car, motor, bus, truck
    classes_person: Tuple[int, ...] = (0,)


@dataclass
class SystemConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    hough: HoughConfig = field(default_factory=HoughConfig)
    parking: ParkingThresholds = field(default_factory=ParkingThresholds)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)

    # ROS 관련
    ros_camera_topic: str = "/camera/color/image_raw"
    ros_cmd_vel_topic: str = "/cmd_vel"
    ros_node_name: str = "parking_robot_node"

    # UI / 로그
    enable_ui: bool = True
    enable_file_logging: bool = True
    log_dir: str = "./logs"
    snapshot_dir: str = "./snapshots"
    log_level: int = logging.DEBUG

    # 시뮬레이션/실로봇 모드
    use_ros: bool = True
    use_simulator: bool = True  # True: Gazebo, False: 실로봇


# ---------------------------------------------------------------------------
# 유틸 함수
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def now_str() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def line_angle_deg(x1, y1, x2, y2) -> float:
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle


def line_length(x1, y1, x2, y2) -> float:
    return math.hypot(x2 - x1, y2 - y1)


# ---------------------------------------------------------------------------
# YOLO 기반 객체 검출 모듈
# ---------------------------------------------------------------------------

class YoloDetector:
    def __init__(self, cfg: YOLOConfig):
        self.cfg = cfg
        self.model = None
        if YOLO_AVAILABLE and os.path.exists(cfg.model_path):
            self.model = YOLO(cfg.model_path)
            logging.info(f"[YOLO] Loaded model from {cfg.model_path}")
        else:
            logging.warning("[YOLO] ultralytics.YOLO 사용 불가 또는 모델 파일 없음. "
                            "더미 검출기로 동작합니다.")

    def detect(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        이미지에서 객체 검출 수행.
        반환: [{'cls': int, 'conf': float, 'xyxy': (x1,y1,x2,y2)} ...]
        """
        if self.model is None:
            # 더미 결과 (테스트용)
            h, w, _ = img_bgr.shape
            dummy_box = {
                "cls": 2,
                "conf": 0.9,
                "xyxy": (int(w * 0.4), int(h * 0.4),
                         int(w * 0.6), int(h * 0.7))
            }
            return [dummy_box]

        results = self.model(
            img_bgr,
            conf=self.cfg.conf_thres,
            iou=self.cfg.iou_thres,
            device=self.cfg.device,
            verbose=False
        )

        detections: List[Dict[str, Any]] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                detections.append({
                    "cls": cls_id,
                    "conf": conf,
                    "xyxy": (x1, y1, x2, y2)
                })
        return detections

    def filter_obstacles(self, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """차량/보행자 등 주차 위험 장애물만 필터링."""
        vehicle_cls = set(self.cfg.classes_vehicle)
        person_cls = set(self.cfg.classes_person)
        filtered = [
            d for d in dets
            if d["cls"] in vehicle_cls.union(person_cls)
        ]
        return filtered


# ---------------------------------------------------------------------------
# 주차선 인식 (IPM + Hough)
# ---------------------------------------------------------------------------

class LineDetector:
    def __init__(self, cam_cfg: CameraConfig, hough_cfg: HoughConfig):
        self.cam_cfg = cam_cfg
        self.hough_cfg = hough_cfg
        self.M_ipm = cv2.getPerspectiveTransform(
            cam_cfg.src_points, cam_cfg.dst_points
        )

    def ipm(self, img_bgr: np.ndarray) -> np.ndarray:
        """투영 변환 (Top-view, IPM)."""
        dst_size = (int(self.cam_cfg.dst_points[1][0]),
                    int(self.cam_cfg.dst_points[2][1]))
        top = cv2.warpPerspective(img_bgr, self.M_ipm, dst_size)
        return top

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """주간/야간 등 환경 프로파일 적용은 호출부에서 결정."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # 동적 이진화 (조도 변화 대응)
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,  # blockSize
            5    # C
        )
        return binary

    def detect_lines(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """확률적 허프변환으로 주차선 후보 검출."""
        lines_p = cv2.HoughLinesP(
            binary,
            rho=self.hough_cfg.rho,
            theta=self.hough_cfg.theta,
            threshold=self.hough_cfg.threshold,
            minLineLength=self.hough_cfg.min_line_length,
            maxLineGap=self.hough_cfg.max_line_gap
        )
        line_list: List[Tuple[int, int, int, int]] = []
        if lines_p is not None:
            for l in lines_p:
                x1, y1, x2, y2 = l[0]
                line_list.append((x1, y1, x2, y2))
        return line_list


# ---------------------------------------------------------------------------
# 주차 가능성 평가 (3단계 판정 + Risk Score)
# ---------------------------------------------------------------------------

@dataclass
class ParkingSlot:
    left_line: Tuple[int, int, int, int]
    right_line: Tuple[int, int, int, int]
    width_m: float
    parallel_deg: float
    risk_score: float
    decision: ParkingDecision


class ParkingEvaluator:
    """
    - 주차선 쌍 매칭
    - 폭/평행도 계산
    - 장애물/엣지케이스 반영한 risk score 산출
    - 3단계 판정 (가능/보류/불가)
    """

    def __init__(self, cfg: ParkingThresholds):
        self.cfg = cfg

    def _pair_lines(self, lines: List[Tuple[int, int, int, int]]) \
            -> List[Tuple[Tuple[int, int, int, int],
                          Tuple[int, int, int, int]]]:
        pairs = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                l1 = lines[i]
                l2 = lines[j]
                # 기울기가 비교적 비슷한 라인만 후보로
                ang1 = line_angle_deg(*l1)
                ang2 = line_angle_deg(*l2)
                if abs(ang1 - ang2) > 20:
                    continue
                pairs.append((l1, l2))
        return pairs

    def _estimate_width_meter(self,
                              l1: Tuple[int, int, int, int],
                              l2: Tuple[int, int, int, int],
                              px_to_m: float = 0.02) -> float:
        """픽셀 간격 -> 대략적인 실제 폭(m) 추정 (단순 비례)."""
        x1 = (l1[0] + l1[2]) / 2.0
        x2 = (l2[0] + l2[2]) / 2.0
        width_px = abs(x2 - x1)
        return width_px * px_to_m

    def _calc_parallel_deg(self, l1, l2) -> float:
        ang1 = line_angle_deg(*l1)
        ang2 = line_angle_deg(*l2)
        return abs(ang1 - ang2)

    def _calc_obstacle_risk(self,
                            slot_center: Tuple[int, int],
                            obstacles: List[Dict[str, Any]]) -> float:
        """
        slot_center와 장애물 사이의 거리가 가까울수록 높은 risk.
        거리 기준은 단순 픽셀 거리 -> 상대값으로 사용.
        """
        if not obstacles:
            return 0.0
        min_dist = float("inf")
        cx, cy = slot_center
        for ob in obstacles:
            x1, y1, x2, y2 = ob["xyxy"]
            ox = (x1 + x2) / 2.0
            oy = (y1 + y2) / 2.0
            d = math.hypot(ox - cx, oy - cy)
            min_dist = min(min_dist, d)
        # 거리 짧을수록 위험: 간단히 0~1 정규화
        # (100px 이내면 risk ~ 1, 400px 이상이면 risk ~ 0)
        max_d = 400.0
        min_d_cap = 100.0
        d_clamp = max(min_d_cap, min(max_d, min_dist))
        risk = 1.0 - (d_clamp - min_d_cap) / (max_d - min_d_cap)
        return risk

    def _decision_from_risk(self, risk: float) -> ParkingDecision:
        if risk < self.cfg.risk_hold:
            return ParkingDecision.POSSIBLE
        elif risk < self.cfg.risk_impossible:
            return ParkingDecision.HOLD
        else:
            return ParkingDecision.IMPOSSIBLE

    def evaluate(self,
                 lines: List[Tuple[int, int, int, int]],
                 obstacles: List[Dict[str, Any]],
                 ipm_shape: Tuple[int, int]) -> List[ParkingSlot]:
        """
        주차선 목록과 장애물 정보를 바탕으로 가능한 주차 슬롯 후보 목록 반환.
        """
        slots: List[ParkingSlot] = []
        if len(lines) < 2:
            return slots

        pairs = self._pair_lines(lines)
        h, w = ipm_shape[:2]
        for l1, l2 in pairs:
            width_m = self._estimate_width_meter(l1, l2)
            if not (self.cfg.min_width_m <= width_m <= self.cfg.max_width_m):
                continue

            parallel_deg = self._calc_parallel_deg(l1, l2)
            if parallel_deg > self.cfg.max_parallel_deg:
                continue

            # 주차 슬롯 중앙 좌표(단순 계산)
            cx = int((l1[0] + l1[2] + l2[0] + l2[2]) / 4.0)
            cy = int(h * 0.7)  # 대략 로봇 기준 앞쪽 영역
            slot_center = (cx, cy)

            obstacle_risk = self._calc_obstacle_risk(slot_center, obstacles)
            # 여기서는 장애물 리스크만 쓰지만, 향후 라인 품질, 텍스처 등도 합산 가능
            risk_score = obstacle_risk

            decision = self._decision_from_risk(risk_score)

            slots.append(ParkingSlot(
                left_line=l1,
                right_line=l2,
                width_m=width_m,
                parallel_deg=parallel_deg,
                risk_score=risk_score,
                decision=decision
            ))
        return slots


# ---------------------------------------------------------------------------
# ROS / Gazebo 연동 및 제어
# ---------------------------------------------------------------------------

class RosGazeboBridge:
    """
    - 카메라 토픽 구독
    - cmd_vel로 주행 명령 발행
    - (필요 시) Gazebo 서비스 호출 등 확장 가능
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.bridge = CvBridge() if ROS_AVAILABLE else None
        self.cmd_pub = None
        self.image_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
        self.running = False

    def init_ros(self):
        if not ROS_AVAILABLE:
            logging.warning("[ROS] ROS Python 패키지를 찾을 수 없습니다. "
                            "ROS 기능은 비활성화됩니다.")
            return

        rospy.init_node(self.cfg.ros_node_name, anonymous=True)
        self.cmd_pub = rospy.Publisher(
            self.cfg.ros_cmd_vel_topic,
            Twist,
            queue_size=1
        )
        rospy.Subscriber(
            self.cfg.ros_camera_topic,
            Image,
            self._camera_callback,
            queue_size=1
        )
        logging.info(f"[ROS] Subscribed camera: {self.cfg.ros_camera_topic}")
        logging.info(f"[ROS] Publishing cmd_vel: {self.cfg.ros_cmd_vel_topic}")

    def _camera_callback(self, msg: Image):
        if not ROS_AVAILABLE or self.bridge is None:
            return
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if not self.image_queue.full():
            self.image_queue.put(cv_img)

    def get_latest_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            frame = self.image_queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None

    def send_stop(self):
        if not ROS_AVAILABLE or self.cmd_pub is None:
            return
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def send_parking_command(self, decision: ParkingDecision):
        """
        판정 결과에 따른 단순 제어 예시.
        실제 프로젝트에서는 Path Planner와 연동해서 사용.
        """
        if not ROS_AVAILABLE or self.cmd_pub is None:
            return
        twist = Twist()
        if decision == ParkingDecision.POSSIBLE:
            twist.linear.x = 0.2  # 진입
            twist.angular.z = 0.0
        elif decision == ParkingDecision.HOLD:
            twist.linear.x = 0.0  # 정지 후 재탐색
            twist.angular.z = 0.2
        else:  # IMPOSSIBLE
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        self.cmd_pub.publish(twist)


# ---------------------------------------------------------------------------
# UI 및 로그 관리
# ---------------------------------------------------------------------------

class UILogger:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        if cfg.enable_file_logging:
            ensure_dir(cfg.log_dir)
        if cfg.enable_ui:
            ensure_dir(cfg.snapshot_dir)
        logging.basicConfig(
            level=cfg.log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    os.path.join(cfg.log_dir, f"parking_{now_str()}.log"),
                    encoding="utf-8"
                ) if cfg.enable_file_logging else logging.NullHandler()
            ]
        )

    def draw_detections(self,
                        img: np.ndarray,
                        obstacles: List[Dict[str, Any]]) -> np.ndarray:
        for det in obstacles:
            x1, y1, x2, y2 = det["xyxy"]
            conf = det.get("conf", 0.0)
            cls_id = det.get("cls", -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                img,
                f"id{cls_id}:{conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
        return img

    def draw_parking_slots(self,
                           img: np.ndarray,
                           slots: List[ParkingSlot]) -> np.ndarray:
        for s in slots:
            color = (0, 255, 0)   # GREEN
            if s.decision == ParkingDecision.HOLD:
                color = (0, 255, 255)  # YELLOW
            elif s.decision == ParkingDecision.IMPOSSIBLE:
                color = (0, 0, 255)  # RED
            x1, y1, x2, y2 = s.left_line
            cv2.line(img, (x1, y1), (x2, y2), color, 2)
            x1, y1, x2, y2 = s.right_line
            cv2.line(img, (x1, y1), (x2, y2), color, 2)

            text = f"{s.decision.name} W:{s.width_m:.2f}m R:{s.risk_score:.2f}"
            cv2.putText(
                img,
                text,
                (min(s.left_line[0], s.right_line[0]),
                 min(s.left_line[1], s.right_line[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        return img

    def show(self, win_name: str, img: np.ndarray, delay: int = 1):
        if not self.cfg.enable_ui:
            return
        cv2.imshow(win_name, img)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("s"):
            # snapshot 저장
            filename = os.path.join(
                self.cfg.snapshot_dir,
                f"{now_str()}.png"
            )
            cv2.imwrite(filename, img)
            logging.info(f"[UI] Snapshot saved: {filename}")
        elif key == ord("q"):
            raise KeyboardInterrupt

    def log_decision(self,
                     decision: ParkingDecision,
                     slots: List[ParkingSlot]):
        if not slots:
            logging.info("[DECISION] No valid parking slot detected.")
            return
        best = sorted(slots, key=lambda x: x.risk_score)[0]
        logging.info(
            f"[DECISION] {decision.name} "
            f"(best risk={best.risk_score:.3f}, "
            f"width={best.width_m:.2f}m, parallel={best.parallel_deg:.2f}°)"
        )


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

class ParkingRobotSystem:
    def __init__(self, cfg: Optional[SystemConfig] = None):
        self.cfg = cfg or SystemConfig()
        self.detector = YoloDetector(self.cfg.yolo)
        self.line_detector = LineDetector(self.cfg.camera, self.cfg.hough)
        self.evaluator = ParkingEvaluator(self.cfg.parking)
        self.ui_logger = UILogger(self.cfg)
        self.bridge = RosGazeboBridge(self.cfg)

    def init(self):
        logging.info("[SYSTEM] Initializing ParkingRobotSystem...")
        if self.cfg.use_ros:
            self.bridge.init_ros()
        logging.info("[SYSTEM] Initialization done.")

    def _get_frame(self) -> Optional[np.ndarray]:
        """
        ROS 또는 웹캠/동영상에서 프레임 가져오기.
        실제 제출 시에는 하나만 선택해서 사용.
        """
        if self.cfg.use_ros and ROS_AVAILABLE:
            return self.bridge.get_latest_frame()
        else:
            # 예시용: 기본 카메라에서 읽기 (테스트용)
            if not hasattr(self, "_cap"):
                self._cap = cv2.VideoCapture(0)
                if not self._cap.isOpened():
                    logging.error("[SYSTEM] Cannot open local camera.")
                    return None
            ret, frame = self._cap.read()
            if not ret:
                return None
            return frame

    def process_once(self) -> bool:
        """
        파이프라인 한 사이클 수행.
        반환값: 계속 진행(True) / 중단(False)
        """
        frame = self._get_frame()
        if frame is None:
            time.sleep(0.01)
            return True  # ROS 이미지 안 들어올 수도 있으니 계속 대기

        # 1) YOLO 객체 검출
        dets = self.detector.detect(frame)
        obstacles = self.detector.filter_obstacles(dets)

        # 2) IPM + 주차선 인식
        ipm_img = self.line_detector.ipm(frame)
        binary = self.line_detector.preprocess(ipm_img)
        lines = self.line_detector.detect_lines(binary)

        # 3) 3단계 주차 가능성 평가
        slots = self.evaluator.evaluate(lines, obstacles, ipm_img.shape)
        if slots:
            # 리스크 가장 낮은 슬롯 기준으로 최종 판정
            best_slot = sorted(slots, key=lambda s: s.risk_score)[0]
            decision = best_slot.decision
        else:
            decision = ParkingDecision.IMPOSSIBLE

        # 4) ROS 제어 명령
        if self.cfg.use_ros:
            self.bridge.send_parking_command(decision)

        # 5) UI 및 로그
        vis = frame.copy()
        vis = self.ui_logger.draw_detections(vis, obstacles)
        vis = self.ui_logger.draw_parking_slots(vis, slots)
        cv2.putText(
            vis,
            f"DECISION: {decision.name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if decision == ParkingDecision.POSSIBLE
            else (0, 255, 255) if decision == ParkingDecision.HOLD
            else (0, 0, 255),
            2
        )
        self.ui_logger.show("ParkingRobot", vis)
        self.ui_logger.log_decision(decision, slots)

        return True

    def run(self):
        self.init()
        logging.info("[SYSTEM] Start main loop.")
        try:
            while True:
                if not self.process_once():
                    break
                if self.cfg.use_ros and ROS_AVAILABLE and rospy.is_shutdown():
                    break
        except KeyboardInterrupt:
            logging.info("[SYSTEM] KeyboardInterrupt received, shutting down...")
        finally:
            if hasattr(self, "_cap"):
                self._cap.release()
            cv2.destroyAllWindows()
            if self.cfg.use_ros:
                self.bridge.send_stop()
        logging.info("[SYSTEM] Stopped.")


# ---------------------------------------------------------------------------
# 엔트리 포인트
# ---------------------------------------------------------------------------

def main():
    # 필요하면 config를 JSON 등에서 불러와서 생성해도 됨
    cfg = SystemConfig()
    system = ParkingRobotSystem(cfg)
    system.run()


if __name__ == "__main__":
    main()
