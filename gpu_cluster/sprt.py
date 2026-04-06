import numpy as np
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger("SPRT_Filter")

class SPRTFilter:
    def __init__(self, sota_threshold: float, confidence_level: float = 0.95, max_steps: int = 10000):
        """
        Initializes the Sequential Probability Ratio Test (SPRT) filter for early stopping.

        Args:
            sota_threshold (float): The target BPB value to beat (e.g., 1.0).
            confidence_level (float): The required statistical confidence to trigger an early abort.
            max_steps (int): Total expected training steps (derived from 10 min wall-clock limit).
        """
        self.sota_threshold = sota_threshold
        self.confidence_level = confidence_level
        self.max_steps = max_steps
        self.loss_history = []
        self.step_history = []

    def _power_law_curve(self, t, a, b, c):
        """
        Power-law curve formula: L(t) = a * t^(-b) + c
        """
        # Add a small epsilon to t to avoid division by zero
        return a * np.power(t + 1e-5, -b) + c

    def update_and_check(self, current_step: int, current_loss: float) -> bool:
        """
        Updates the history with the new loss metric and evaluates if the run should be aborted.
        Returns True if the run should be ABORTED, False otherwise.
        """
        self.step_history.append(current_step)
        self.loss_history.append(current_loss)

        # Need a minimum number of points to fit the curve reliably
        if len(self.loss_history) < 5:
            return False

        # Plateau Detection (If loss hasn't improved in 5 checks, kill it)
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-5:]
            if max(recent_losses) - min(recent_losses) < 0.001 and min(recent_losses) > self.sota_threshold:
                logger.info(f"SPRT ABORT: Loss plateau detected at {min(recent_losses):.4f} > {self.sota_threshold:.4f}")
                return True

        t_data = np.array(self.step_history)
        L_data = np.array(self.loss_history)

        try:
            # Fit the power-law curve to the empirical loss data
            # Bounds: a > 0, b > 0, c > 0
            popt, pcov = curve_fit(
                self._power_law_curve,
                t_data,
                L_data,
                bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
                maxfev=5000
            )

            a, b, c = popt

            # Extrapolate to the end of training (max_steps)
            projected_final_loss = self._power_law_curve(self.max_steps, a, b, c)

            # Estimate standard error of the offset (c) from the covariance matrix
            # This provides a basic confidence interval around the projection
            perr = np.sqrt(np.diag(pcov))
            c_std_err = perr[2]

            # Calculate the lower bound of our projection based on confidence level
            # (Assuming normal distribution of error, 95% is approx 1.96 standard errors)
            # If the best-case (lower bound) is still worse than SOTA + 5% margin, we abort
            lower_bound_projection = projected_final_loss - (1.96 * c_std_err)

            margin = 0.05 # 5% margin

            if lower_bound_projection > self.sota_threshold * (1 + margin):
                logger.info(f"SPRT ABORT triggered at step {current_step}.")
                logger.info(f"Projected Best-Case BPB: {lower_bound_projection:.4f} > Threshold: {self.sota_threshold:.4f}")
                return True

        except Exception as e:
            # If curve fitting fails (e.g., Optimal parameters not found), we don't abort to be safe
            logger.debug(f"Curve fitting failed at step {current_step}: {e}")

        return False
