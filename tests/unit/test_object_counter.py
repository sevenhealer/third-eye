from src.object_counting.counter import ObjectCountAggregator


def test_bucket_stores_max_simultaneous():
    agg = ObjectCountAggregator("cam0", "bedroom", bucket_seconds=10)
    agg.observe({"person": 1, "bottle": 2}, timestamp=100.0)
    agg.observe({"person": 3}, timestamp=104.0)
    agg.observe({"person": 2, "bottle": 1}, timestamp=109.0)
    buckets = agg.pop_closed(now=115.0)
    by_class = {b.object_class: b.count for b in buckets}
    assert by_class == {"person": 3, "bottle": 2}
    assert all(b.camera_id == "cam0" and b.zone_id == "bedroom" for b in buckets)


def test_buckets_roll_over_on_boundary():
    agg = ObjectCountAggregator("cam0", None, bucket_seconds=10)
    agg.observe({"person": 2}, timestamp=100.0)
    agg.observe({"person": 1}, timestamp=111.0)   # next bucket
    buckets = agg.pop_closed(now=111.0)
    assert len(buckets) == 1
    assert buckets[0].count == 2
    # second bucket still open
    assert agg.pop_closed(now=115.0) == []
    assert len(agg.pop_closed(now=121.0)) == 1


def test_pop_without_due_buckets_is_empty():
    agg = ObjectCountAggregator("cam0", "z", bucket_seconds=10)
    agg.observe({"person": 1}, timestamp=100.0)
    assert agg.pop_closed(now=105.0) == []
