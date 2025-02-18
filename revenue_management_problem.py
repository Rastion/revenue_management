import math
import random
from qubots.base_problem import BaseProblem

def exponential_sample(rate_param=1.0):
    u = random.random()
    return math.log(1 - u) / (-rate_param)

def gamma_sample(scale_param=1.0):
    # For k=1, a gamma sample is equivalent to an exponential sample.
    return exponential_sample(scale_param)

class RevenueManagementProblem(BaseProblem):
    """
    Revenue Management Problem

    A businessman must decide the total number of units to purchase at the
    beginning of a time horizon and the amount to reserve for later periods,
    so as to maximize revenue. The time horizon is divided into 3 periods.
    
    Parameters:
      - The decision variables are three integers in [0, 100]:
           [purchase, reserve_period2, reserve_period3]
        where feasibility requires:
           reserve_period2 ≤ purchase   and   reserve_period3 ≤ reserve_period2.
    
    Simulation details:
      - Prices: [100, 300, 400] for periods 1, 2, and 3.
      - Mean demands: [50, 20, 30] for periods 1, 2, and 3.
      - Purchase cost per unit: 80.
      - Demand in period t is modeled as:
            Dₜ = μₜ * X * Yₜ
        where X is sampled from a gamma distribution (with shape 1, scale 1)
        and Yₜ from an exponential distribution with rate 1.
      - To obtain a robust revenue estimate, the simulation is run over a large
        number of iterations (by default, 1e6).
    
    The mean revenue is computed as the average sales revenue (over the three
    periods) minus the purchase cost. Infeasible solutions (where the reservation
    ordering is violated) are heavily penalized.
    
    Note: Since the goal is to maximize revenue, the evaluation function returns
    the mean revenue (and the problem_config.json will indicate a maximization objective).
    """

    def __init__(self, seed):
        self.nb_periods = 3
        self.prices = [100, 300, 400]
        self.mean_demands = [50, 20, 30]
        self.purchase_price = 80
        self.nb_simulations = int(1e2)  # Number of Monte Carlo simulations
        self.seed = seed
        # (For reference, a previously evaluated point is [100, 50, 30] with value 4740.99)

    def evaluate_solution(self, candidate) -> float:
        """
        Evaluates a candidate solution.
        
        Parameters:
          candidate: A list of three integers [purchase, reserve_period2, reserve_period3].
        
        Returns:
          The estimated mean revenue (to be maximized). Infeasible solutions are penalized.
        """
        if len(candidate) != self.nb_periods:
            raise ValueError(f"Candidate solution must have {self.nb_periods} values.")
        
        purchase = candidate[0]
        # For periods 1 and 2, reservations are provided; for period 3 no reservation is needed.
        reserves = candidate[1:] + [0]
        
        # Enforce feasibility: reservations must be nonincreasing.
        if candidate[1] > candidate[0] or candidate[2] > candidate[1]:
            return -1e9  # Heavy penalty for infeasibility

        # Set the seed for reproducibility.
        #random.seed(self.seed)
        sum_profit = 0.0

        for _ in range(self.nb_simulations):
            # Sample X from a gamma distribution (here equivalent to an exponential sample)
            X = gamma_sample()
            # Sample Y for each period (exponential with rate=1)
            Y = [exponential_sample() for _ in range(self.nb_periods)]
            remaining = purchase
            for j in range(self.nb_periods):
                # Compute simulated demand for period j.
                demand = int(self.mean_demands[j] * X * Y[j])
                # Only sell what is available after reserving for future periods.
                available = max(remaining - reserves[j], 0)
                units_sold = min(available, demand)
                remaining -= units_sold
                sum_profit += self.prices[j] * units_sold

        mean_profit = sum_profit / self.nb_simulations
        mean_revenue = mean_profit - self.purchase_price * purchase
        return mean_revenue

    def random_solution(self):
        """
        Generates a random feasible solution.
        
        The method ensures the reservation ordering by sampling sequentially.
        """
        purchase = random.randint(0, 100)
        reserve_p2 = random.randint(0, purchase)
        reserve_p3 = random.randint(0, reserve_p2)
        return [purchase, reserve_p2, reserve_p3]
