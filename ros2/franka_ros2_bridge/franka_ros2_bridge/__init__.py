"""Reusable ROS 2 bridge for controlling a Franka robot through frankx.

Package layout:
  - ros/      message receive / send helpers
  - core/  shared types, safety checks, command queue
  - control/ robot backends (frankx today; extend here later)
"""
