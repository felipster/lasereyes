"""
PIDController: Simple PID controller for servo axis control.
"""


class PIDController:
    """Simple PID controller for servo axis control."""
    
    def __init__(self, kp: float, ki: float, kd: float):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
    
    def update(self, error: float, dt: float) -> float:
        """
        Compute PID output.
        
        Args:
            error: Current error signal
            dt: Time step
            
        Returns:
            Control output (servo angle adjustment)
        """
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        return output
    
    def reset(self):
        """Reset controller state."""
        self.prev_error = 0
        self.integral = 0
