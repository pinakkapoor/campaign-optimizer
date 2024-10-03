"""
budget optimizer across marketing channels

given a set of channels with fitted ROI curves and a total budget,
find the allocation that maximizes total revenue. uses equi-marginal
principle — keep reallocating until marginal ROI is equal across channels.
"""

import numpy as np
from scipy.optimize import minimize

from campaign_optimizer.models.roi_curve import ROICurve


class CampaignOptimizer:
    """
    optimizes budget allocation across marketing channels.

    usage:
        opt = CampaignOptimizer(total_budget=100000)
        opt.add_channel('paid_search', spend_history, revenue_history)
        opt.add_channel('social', spend_history2, revenue_history2)
        result = opt.optimize()
    """

    def __init__(self, total_budget, curve_type='log'):
        self.total_budget = total_budget
        self.curve_type = curve_type
        self.channels = {}

    def add_channel(self, name, spend_history, revenue_history,
                    min_spend=0, max_spend=None):
        """
        add a channel with historical data and optional constraints.

        min_spend: minimum budget (contractual obligations, etc)
        max_spend: maximum budget (capacity constraints)
        """
        curve = ROICurve(curve_type=self.curve_type)
        curve.fit(spend_history, revenue_history)

        self.channels[name] = {
            'curve': curve,
            'min_spend': min_spend,
            'max_spend': max_spend or self.total_budget,
            'historical_spend': np.array(spend_history),
            'historical_revenue': np.array(revenue_history),
        }

    def optimize(self):
        """
        find the revenue-maximizing allocation.

        uses scipy minimize (SLSQP) with budget constraint and
        per-channel min/max bounds.
        """
        channel_names = list(self.channels.keys())
        n = len(channel_names)

        if n == 0:
            raise ValueError("add at least one channel before optimizing")

        # objective: negative total revenue (because we minimize)
        def neg_revenue(allocations):
            total = 0
            for i, name in enumerate(channel_names):
                total += self.channels[name]['curve'].predict(allocations[i])
            return -total

        # constraint: allocations sum to total budget
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_budget}

        # bounds per channel
        bounds = []
        for name in channel_names:
            ch = self.channels[name]
            bounds.append((ch['min_spend'], ch['max_spend']))

        # start with equal allocation
        x0 = np.full(n, self.total_budget / n)

        result = minimize(neg_revenue, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        if not result.success:
            raise RuntimeError(f"optimization failed: {result.message}")

        # build results
        allocation = {}
        total_predicted_revenue = 0
        for i, name in enumerate(channel_names):
            spend = result.x[i]
            revenue = self.channels[name]['curve'].predict(spend)
            marginal = self.channels[name]['curve'].marginal_roi(spend)
            total_predicted_revenue += revenue

            allocation[name] = {
                'spend': round(float(spend), 2),
                'predicted_revenue': round(float(revenue), 2),
                'marginal_roi': round(float(marginal), 4),
            }

        # compare to current (average historical) allocation
        current_revenue = sum(
            ch['historical_revenue'].mean()
            for ch in self.channels.values()
        )

        return {
            'allocation': allocation,
            'total_budget': self.total_budget,
            'predicted_revenue': round(total_predicted_revenue, 2),
            'current_revenue': round(float(current_revenue), 2),
            'improvement': round(total_predicted_revenue - float(current_revenue), 2),
            'improvement_pct': round(
                (total_predicted_revenue - float(current_revenue)) / float(current_revenue) * 100,
                1
            ) if current_revenue > 0 else None,
        }

    def rebalance(self, current_allocation, budget_change=0):
        """
        suggest reallocation given current spend levels and optional budget change.
        useful for mid-flight optimization.
        """
        self.total_budget = sum(current_allocation.values()) + budget_change
        optimized = self.optimize()

        changes = {}
        for name in current_allocation:
            if name in optimized['allocation']:
                current = current_allocation[name]
                suggested = optimized['allocation'][name]['spend']
                changes[name] = {
                    'current': current,
                    'suggested': suggested,
                    'change': round(suggested - current, 2),
                    'change_pct': round((suggested - current) / current * 100, 1) if current > 0 else None,
                }

        return {
            'changes': changes,
            'optimized': optimized,
        }
