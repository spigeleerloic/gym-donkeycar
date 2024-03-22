class PIDController:
    def __init__(self, kp, ki, kd, min_output=-5.0, max_output=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.min_output = min_output
        self.max_output = max_output

        self.error_sum = 0.0
        self.prev_error = 0.0
    
    def get_kp(self):
        return self.kp
    
    def get_ki(self):
        return self.ki
    
    def get_kd(self):
        return self.kd
    
    def get_min_output(self):
        return self.min_output
    
    def get_max_output(self):
        return self.max_output
    
    def set_kp(self, kp):
        self.kp = kp
    
    def set_ki(self, ki):
        self.ki = ki

    def set_kd(self, kd):
        self.kd = kd
    
    def set_min_output(self, min_output):
        self.min_output = min_output
    
    def set_max_output(self, max_output):
        self.max_output = max_output
    

    def update(self, expected_value, observed_value):
        """
        update the PID controller with the observed value and the expected value
        output is related to the proportional, integral and derivative terms
        The output belong to the range of min_output and max_output

        :param expected_value: the expected value
        :param observed_value: the observed value

        :return: the output of the PID controller
        """
        error = expected_value - observed_value
        self.error_sum += error

        marginal_error = error - self.prev_error
        # proportional to error term, integral to sum of errors, derivative to change in error
        output = self.kp * error + self.ki * self.error_sum + self.kd * marginal_error
        # clamp output to the range of min_output and max_output
        output = max(self.min_output, min(self.max_output, output))

        self.prev_error = error
        return output
    
    