from typing import Any, Dict, Tuple
from PIDController import PIDController

def compute_distance_from_center_line(info: Dict[str, Any]) -> float:
    """
    Compute the distance from the center line.
    
    Args:
        info (Dict[str, Any]): information about the environment
    
    Returns:
        float: distance from the center line
    """
    return info["distance_to_middle_line"]

def compute_forward_speed(info: Dict[str, Any]) -> float:
    """
    Compute the forward speed.
    
    Args:
        info (Dict[str, Any]): information about the environment
    
    Returns:
        float: forward speed
    """
    return info["forward_vel"]

def ultimate_gain_to_controller(ultimate_gain: float, oscillation_period: float, min_output : float, max_output : float) -> Tuple[float, float]:
    """
    Convert the ultimate gain to the PID controller parameters.
    
    Args:
        ultimate_gain (float): ultimate gain
        period (float): period
    
    Returns:
        Tuple[float, float]: proportional and integral gain
    """
    kp = 0.6 * ultimate_gain
    ki = 1.2 * ultimate_gain / oscillation_period
    kd = 0.075 * ultimate_gain * oscillation_period
    return PIDController(kp=kp, ki=ki, kd=kd, min_output=min_output, max_output=max_output)

