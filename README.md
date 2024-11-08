# campaign-optimizer

campaign budget optimization across marketing channels. fits ROI curves to historical data, then finds the allocation that maximizes total revenue.

most campaign analytics lives in spreadsheets. someone eyeballs the ROAS by channel and moves money around. this does it properly — fits diminishing returns curves to each channel and uses constrained optimization to find the best allocation.

## how it works

1. feed it historical spend/revenue data per channel
2. it fits ROI curves (log or hill/saturation curves)
3. optimizer finds the allocation that maximizes total revenue, respecting per-channel min/max constraints
4. supports mid-flight rebalancing

```python
from campaign_optimizer import CampaignOptimizer

opt = CampaignOptimizer(total_budget=100000)

opt.add_channel('paid_search', spend_history, revenue_history)
opt.add_channel('social', spend_history2, revenue_history2, min_spend=5000)
opt.add_channel('display', spend_history3, revenue_history3)

result = opt.optimize()
print(result['allocation'])
# {'paid_search': {'spend': 52000, 'predicted_revenue': 84000, 'marginal_roi': 0.82},
#  'social': {'spend': 31000, 'predicted_revenue': 41000, 'marginal_roi': 0.81},
#  'display': {'spend': 17000, 'predicted_revenue': 19000, 'marginal_roi': 0.80}}
```

## roi curves

two curve types:
- **log**: `revenue = a * ln(spend + 1) + b`. simple, works well with limited data
- **hill**: `revenue = max * spend^k / (half_sat^k + spend^k)`. better for channels with clear saturation

use hill if you have enough data points (10+), log otherwise.

## rebalancing

for mid-flight adjustments:

```python
current = {'paid_search': 40000, 'social': 35000, 'display': 25000}
changes = opt.rebalance(current)
# tells you how to shift budget for remaining campaign duration
```

## install

```
pip install -e .
```

## limitations

- assumes historical ROI curves are stable (they're not, but it's a reasonable starting point)
- doesn't account for cross-channel effects (paid search + social together > sum of parts)
- small data = noisy curves. need at least 5-6 data points per channel for log, more for hill
