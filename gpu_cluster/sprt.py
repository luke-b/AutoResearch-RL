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

            # Simple confidence interval check using covariance matrix
            # If the projected loss + confidence margin is STILL strictly worse than SOTA, we abort.
            # (Note: This is a simplified SPRT implementation adapted for deterministic power-law bounds)

            # Estimate standard error of the projection (simplified)
            perr = np.sqrt(np.diag(pcov))
            # Just look at the offset uncertainty (c) as a very rough bound, or compute full Jacobian.
            # For this MVP, we will assume an abort if the raw projection is significantly higher than SOTA.

            # Let's say if projected_final_loss > SOTA by a margin, we abort.
            # In a true SPRT, we'd use log-likelihood ratios. Here we use the specified power-law extrapolation.
            margin = 0.05 # 5% margin
            if projected_final_loss > self.sota_threshold * (1 + margin):
                logger.info(f"SPRT ABORT triggered at step {current_step}. Projected final BPB: {projected_final_loss:.4f} > Threshold: {self.sota_threshold:.4f}")
                return True

        except Exception as e:
            # If curve fitting fails (e.g., Optimal parameters not found), we don't abort to be safe
            logger.debug(f"Curve fitting failed at step {current_step}: {e}")

        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    filter = SPRTFilter(sota_threshold=0.98, max_steps=1000)

    # Simulate a run that won't make it
    for step in range(50, 600, 50):
        # Fake data that flattens out around 1.1
        fake_loss = 2.0 * (step)**(-0.5) + 1.1
        should_abort = filter.update_and_check(step, fake_loss)
        if should_abort:
            print(f"Aborted early at step {step}!")
            break
