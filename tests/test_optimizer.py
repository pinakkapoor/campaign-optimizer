"""tests for campaign optimizer"""

import numpy as np
from campaign_optimizer import CampaignOptimizer, ROICurve


def test_log_curve_fit():
    """log curve should fit logarithmic data"""
    spend = np.array([100, 500, 1000, 2000, 5000, 10000])
    # revenue roughly follows log(spend)
    revenue = 500 * np.log(spend + 1) + np.random.normal(0, 50, len(spend))

    curve = ROICurve('log')
    curve.fit(spend, revenue)

    pred = curve.predict(np.array([3000]))
    assert 3000 < pred < 5000, f"prediction {pred} seems off"


def test_hill_curve_fit():
    """hill curve should capture saturation"""
    spend = np.array([100, 500, 1000, 2000, 5000, 10000, 20000, 50000])
    # saturating revenue
    revenue = 100000 * spend / (5000 + spend)

    curve = ROICurve('hill')
    curve.fit(spend, revenue)

    # should predict saturation at high spend
    low_pred = curve.predict(np.array([1000]))
    high_pred = curve.predict(np.array([100000]))
    assert high_pred > low_pred
    assert high_pred < 120000, "should saturate"


def test_marginal_roi_decreases():
    """marginal ROI should decrease with spend (diminishing returns)"""
    spend = np.array([100, 500, 1000, 2000, 5000, 10000])
    revenue = 500 * np.log(spend + 1)

    curve = ROICurve('log')
    curve.fit(spend, revenue)

    mroi_low = curve.marginal_roi(np.array([1000]))
    mroi_high = curve.marginal_roi(np.array([10000]))
    assert mroi_low > mroi_high, "marginal ROI should decrease"


def test_optimizer_allocates_budget():
    """optimizer should allocate full budget across channels"""
    np.random.seed(42)

    opt = CampaignOptimizer(total_budget=50000)

    # channel 1: high efficiency
    spend1 = np.array([1000, 3000, 5000, 8000, 12000])
    rev1 = 800 * np.log(spend1 + 1)
    opt.add_channel('paid_search', spend1, rev1)

    # channel 2: lower efficiency
    spend2 = np.array([1000, 3000, 5000, 8000, 12000])
    rev2 = 400 * np.log(spend2 + 1)
    opt.add_channel('display', spend2, rev2)

    result = opt.optimize()

    total_allocated = sum(ch['spend'] for ch in result['allocation'].values())
    assert abs(total_allocated - 50000) < 1, f"budget not fully allocated: {total_allocated}"

    # paid_search should get more budget (higher ROI)
    assert result['allocation']['paid_search']['spend'] > result['allocation']['display']['spend']


def test_min_spend_constraint():
    """channels with min_spend should get at least that much"""
    np.random.seed(42)

    opt = CampaignOptimizer(total_budget=20000)

    spend = np.array([1000, 3000, 5000, 8000, 12000])
    rev = 500 * np.log(spend + 1)

    opt.add_channel('good_channel', spend, rev)
    opt.add_channel('contractual', spend, rev * 0.1, min_spend=5000)

    result = opt.optimize()
    assert result['allocation']['contractual']['spend'] >= 5000


if __name__ == '__main__':
    test_log_curve_fit()
    test_hill_curve_fit()
    test_marginal_roi_decreases()
    test_optimizer_allocates_budget()
    test_min_spend_constraint()
    print("all tests passed")
