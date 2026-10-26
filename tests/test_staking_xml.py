from app.config import StakingConfig
from app.strategy.staking import FixedStaking, XmlMartingaleStaking, build_staking_engine, to_cents


def cfg(**kw):
    c = StakingConfig()
    c.mode = "xml_martingale"
    c.first_stake, c.minimum_stake, c.maximum_stake = 1.0, 0.35, 1000.0
    c.martingale_factor, c.martingale_level, c.martingale_start = 1.5, 5, 1
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_ladder_matches_the_xml_exactly():
    # Stake += Stake*1.5 (x2.5) on each of the first 5 losses; 6th loss -> "Martingale Reset"
    s = XmlMartingaleStaking(cfg())
    assert [s.raw_stake(n) for n in range(7)] == [1.0, 2.5, 6.25, 15.625, 39.0625, 97.65625, 1.0]


def test_stays_at_first_stake_until_a_win_after_the_level_is_exceeded():
    s = XmlMartingaleStaking(cfg())
    assert s.raw_stake(9) == 1.0     # loss count keeps rising after the reset; stake stays at first stake


def test_do_martingale_after():
    s = XmlMartingaleStaking(cfg(martingale_start=3))
    assert s.raw_stake(0) == s.raw_stake(1) == s.raw_stake(2) == 1.0
    assert s.raw_stake(3) == 2.5 and s.raw_stake(4) == 6.25


def test_level_zero_disables_the_ladder():
    s = XmlMartingaleStaking(cfg(martingale_level=0))
    assert all(s.raw_stake(n) == 1.0 for n in range(8))


def test_cap_cents_and_minimum():
    s = XmlMartingaleStaking(cfg(maximum_stake=25.0))
    assert s.stake_for(consecutive_losses=3) == 15.62          # rounded DOWN to cents
    assert s.stake_for(consecutive_losses=5) == 25.0           # capped
    assert to_cents(0.349, 0.35) == 0.35
    assert XmlMartingaleStaking(cfg(first_stake=0.35)).stake_for(0) == 0.35


def test_fixed_and_factory():
    c = cfg()
    assert FixedStaking(c).stake_for(consecutive_losses=4) == 1.0
    c.mode = "fixed"
    assert isinstance(build_staking_engine(c), FixedStaking)
    c.mode = "xml_martingale"
    assert isinstance(build_staking_engine(c), XmlMartingaleStaking)
