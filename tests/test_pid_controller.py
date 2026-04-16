"""
test_pid_controller.py: Unit tests for PIDController
"""

import unittest
from src.pid_controller import PIDController


class TestPIDController(unittest.TestCase):
    """Test cases for PIDController class."""
    
    def test_pid_controller_init(self):
        """Test PIDController initialization."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.2)
        
        self.assertEqual(pid.kp, 1.0)
        self.assertEqual(pid.ki, 0.1)
        self.assertEqual(pid.kd, 0.2)
        self.assertEqual(pid.prev_error, 0)
        self.assertEqual(pid.integral, 0)
    
    def test_pid_proportional_response(self):
        """Test proportional response."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        
        # Error of 1.0 should produce output of 1.0
        output = pid.update(error=1.0, dt=0.01)
        self.assertEqual(output, 1.0)
    
    def test_pid_integral_response(self):
        """Test integral response accumulation."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)
        
        # Constant error should accumulate integral term
        output1 = pid.update(error=1.0, dt=0.01)
        output2 = pid.update(error=1.0, dt=0.01)
        
        # Second output should be larger due to accumulated integral
        self.assertGreater(output2, output1)
    
    def test_pid_derivative_response(self):
        """Test derivative response to changing errors."""
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0)
        
        # First update: error=1.0
        output1 = pid.update(error=1.0, dt=0.01)
        
        # Second update: error=2.0 (increasing, should produce derivative term)
        output2 = pid.update(error=2.0, dt=0.01)
        
        # Derivative term should be negative (error is increasing)
        self.assertGreater(output1, output2)
    
    def test_pid_reset(self):
        """Test PID state reset."""
        pid = PIDController(kp=1.0, ki=1.0, kd=1.0)
        
        # Build up some state
        pid.update(error=1.0, dt=0.01)
        pid.update(error=2.0, dt=0.01)
        
        # Reset
        pid.reset()
        
        # State should be cleared
        self.assertEqual(pid.prev_error, 0)
        self.assertEqual(pid.integral, 0)


if __name__ == '__main__':
    unittest.main()
