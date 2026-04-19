"""IMU fusion package for orientation and auto-labeling."""

from src.imu.fusion import AutoLabel, ComplementaryFilter, IMUAutoLabeler

__all__ = ["ComplementaryFilter", "IMUAutoLabeler", "AutoLabel"]
