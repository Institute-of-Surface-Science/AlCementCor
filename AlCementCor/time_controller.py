from abc import ABC, abstractmethod


class TimeController(ABC):
    """
    Base class for time controllers
    """

    def __init__(self, initial_time_step):
        self.time_step = initial_time_step

    @abstractmethod
    def update(self, error):
        """
        Update the time step based on the given error.
        """
        pass


class BasicTimeController(TimeController):
    """
    Basic time controller that does not change the time step.
    """

    def update(self, error):
        """
        No time step change for BasicTimeController
        """
        pass


class PITimeController(TimeController):
    """
    PI controller for time step adaptation
    """

    def __init__(self, initial_time_step, desired_norm_res, kp=0.1, ki=0.05, min_time_step=None, max_time_step=None):
        super().__init__(initial_time_step)
        self.desired_norm_res = desired_norm_res
        self.kp = kp
        self.ki = ki
        self.integral_error = 0.0
        self.min_time_step = min_time_step if min_time_step is not None else 1e-4 * initial_time_step
        self.max_time_step = max_time_step if max_time_step is not None else 10 * initial_time_step

    def update(self, error):
        """
        Update the time step based on the PI controller strategy.
        """
        error = (error - self.desired_norm_res) / self.desired_norm_res
        self.integral_error += error
        delta_time_step = -self.kp * error * self.time_step - self.ki * self.integral_error * self.time_step
        self.time_step += delta_time_step
        self.time_step = max(min(self.time_step, self.max_time_step), self.min_time_step)  # clamp time step in a
        return self.time_step
