"""TDD: clique-finding (which methods are NOT significantly different, per
Demsar's CD diagram convention -- maximal groups of rank-adjacent methods
within CD of each other, minus any group fully contained in a larger one)
and a smoke test that the actual diagram renders without error.
"""
import os
import tempfile

from cd_plot import find_cliques, plot_cd_diagram


def test_no_cliques_when_all_ranks_far_apart():
    ranks_sorted = [1.0, 3.0, 5.0, 7.0]
    assert find_cliques(ranks_sorted, cd=1.0) == []


def test_single_clique_spanning_all_close_ranks():
    ranks_sorted = [1.0, 1.2, 1.4, 1.5]
    cliques = find_cliques(ranks_sorted, cd=1.0)
    assert cliques == [(0, 3)]


def test_two_separate_cliques_not_merged():
    # indices 0,1 close together; indices 2,3 close together; the two
    # groups themselves are far apart (gap of 5 >> cd=1)
    ranks_sorted = [1.0, 1.5, 6.0, 6.5]
    cliques = find_cliques(ranks_sorted, cd=1.0)
    assert cliques == [(0, 1), (2, 3)]


def test_subset_clique_dropped():
    # 0,1,2 all within cd of each other (span 1.0) -> one clique (0,2);
    # (0,1) and (1,2) are real cliques too but fully contained in (0,2),
    # so must NOT appear separately.
    ranks_sorted = [1.0, 1.5, 2.0]
    cliques = find_cliques(ranks_sorted, cd=1.0)
    assert cliques == [(0, 2)]


def test_plot_smoke():
    avg_ranks = {"A": 1.5, "B": 2.0, "C": 4.0, "D": 4.5, "E": 2.5}
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "smoke.png")
        plot_cd_diagram(avg_ranks, cd=1.2, title="Smoke Test", out_path=out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0


if __name__ == "__main__":
    test_no_cliques_when_all_ranks_far_apart()
    test_single_clique_spanning_all_close_ranks()
    test_two_separate_cliques_not_merged()
    test_subset_clique_dropped()
    test_plot_smoke()
    print("All CD plot tests passed.")
