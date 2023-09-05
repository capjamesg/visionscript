from dataclasses import dataclass


@dataclass
class Pose:
    """
    A pose.
    """

    keypoints: list
    confidence: float
