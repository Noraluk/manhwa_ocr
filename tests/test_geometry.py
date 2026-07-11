from app import geometry as geo


def test_quad_to_box():
    assert geo.quad_to_box([[10, 20], [50, 22], [48, 60], [12, 58]]) == (10, 20, 40, 40)


def test_intersection_and_iou():
    a = (0, 0, 10, 10)
    b = (5, 5, 10, 10)
    assert geo.intersection(a, b) == (5, 5, 5, 5)
    assert abs(geo.iou(a, b) - 25 / 175) < 1e-9
    assert geo.iou((0, 0, 10, 10), (100, 100, 5, 5)) == 0.0


def test_overlap_ratio_full_containment():
    inner = (2, 2, 4, 4)
    outer = (0, 0, 20, 20)
    assert geo.overlap_ratio(inner, outer) == 1.0


def test_union_box():
    assert geo.union_box([(0, 0, 5, 5), (10, 10, 5, 5)]) == (0, 0, 15, 15)


def test_pad_clamps_to_bounds():
    assert geo.pad((2, 2, 4, 4), px=10, w_max=8, h_max=8) == (0, 0, 8, 8)


def test_dedup_boxes_removes_overlaps():
    items = [{"box": (0, 0, 10, 10)}, {"box": (1, 1, 10, 10)}, {"box": (100, 100, 4, 4)}]
    kept = geo.dedup_boxes(items, thr=0.5)
    assert len(kept) == 2
