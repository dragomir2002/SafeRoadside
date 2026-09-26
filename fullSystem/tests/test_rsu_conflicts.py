"""Which vehicle-VRU track pairs the RSU's own pixel collision check flags.

The detector draws a red blob wherever a vehicle's predicted point and a VRU's
come closer than COLLISION_THRESHOLD pixels, but it flattens the points and
forgets whose they were. This recovers the pairs, under the same strict rule.
"""
from rsu_conflicts import conflict_pairs


def test_a_vehicle_whose_prediction_meets_the_vru_is_a_pair_with_its_closest_distance():
    pairs = conflict_pairs(car_pts=[(100, 100), (110, 100)], car_owner=[7, 7],
                           vru_pts=[(120, 100)], vru_owner=[3], threshold=30)
    assert pairs == {(7, 3): 10.0}


def test_a_vehicle_that_stays_away_is_not_a_pair():
    pairs = conflict_pairs(car_pts=[(0, 0)], car_owner=[1],
                           vru_pts=[(500, 500)], vru_owner=[2], threshold=30)
    assert pairs == {}


def test_exactly_the_threshold_is_not_a_conflict_as_in_the_detector():
    pairs = conflict_pairs(car_pts=[(0, 0)], car_owner=[1],
                           vru_pts=[(30, 0)], vru_owner=[2], threshold=30)
    assert pairs == {}


def test_each_pair_is_kept_separately():
    pairs = conflict_pairs(car_pts=[(0, 0), (200, 0)], car_owner=[1, 2],
                           vru_pts=[(5, 0), (200, 4)], vru_owner=[8, 9], threshold=30)
    assert pairs == {(1, 8): 5.0, (2, 9): 4.0}


def test_nothing_predicted_means_no_pairs():
    assert conflict_pairs([], [], [(1, 1)], [4], 30) == {}
    assert conflict_pairs([(1, 1)], [4], [], [], 30) == {}
