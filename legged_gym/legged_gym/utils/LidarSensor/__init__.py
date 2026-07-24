import os

PKG_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
SENSOR_ROOT_DIR = PKG_ROOT_DIR
RESOURCES_DIR = os.path.join(PKG_ROOT_DIR, 'resources')

from .lidar_sensor import LidarSensor
from .sensor_config.lidar_sensor_config import LidarConfig, LidarType

__all__ = ["LidarSensor", "LidarConfig", "LidarType"]
