"""TDD: win-count tie-splitting, the 4-decimal tie rule, scientific-notation
formatting, and the global average-rank listing -- written against small
synthetic 'family.strategy' frames so no CSV fixtures are needed.
"""
import pandas as pd

from generate_summary_tables import avg_rank_tex, fmt_value, win_counts, winners


def test_tie_splits_one_over_n():
    row = pd.Series({"a": 1.0, "b": 1.0, "c": 0.5})
    assert winners(row, higher_is_better=True) == ["a", "b"]


def test_row_sums_equal_dataset_count():
    means = pd.DataFrame({
        "fam1.a": [1.0, 2.0, 3.0], "fam1.b": [2.0, 1.0, 1.0],
        "fam2.a": [0.5, 0.5, 0.5], "fam2.b": [0.4, 0.6, 0.5],
    }, index=["DS1", "DS2", "DS3"])
    table = win_counts(means, families=["fam1", "fam2"], strategies=["a", "b"], higher_is_better=True)
    assert table.loc["fam1"].sum() == 3.0
    assert table.loc["fam2"].sum() == 3.0
    assert table.loc["Total"].sum() == 6.0


def test_best_count_respects_direction():
    means = pd.DataFrame({"fam.a": [1.0], "fam.b": [2.0]}, index=["DS1"])
    hib_true = win_counts(means, families=["fam"], strategies=["a", "b"], higher_is_better=True)
    hib_false = win_counts(means, families=["fam"], strategies=["a", "b"], higher_is_better=False)
    assert hib_true.loc["fam", "b"] == 1.0
    assert hib_false.loc["fam", "a"] == 1.0


def test_tie_rule_uses_four_decimal_rounding():
    # Distinct as raw floats, but equal once rounded to 4 decimals -- the
    # tie rule must use rounding, not exact `==` on the raw values.
    row = pd.Series({"a": 1.00001, "b": 1.00002})
    assert row["a"] != row["b"]
    assert winners(row, higher_is_better=True) == ["a", "b"]


def test_fmt_value_switches_to_scientific():
    assert fmt_value(123.4567) == "123.4567"
    assert fmt_value(0.0) == "0.0000"
    assert fmt_value(0.00005) == r"$5.0000 \times 10^{-5}$"
    assert fmt_value(4.05e8) == r"$4.0500 \times 10^{8}$"


def test_avg_rank_table_lists_all_50_combos_sorted():
    cols = [f"fam{i}.s{j}" for i in range(5) for j in range(10)]
    means = pd.DataFrame([[float(v) for v in range(len(cols))]] * 20, columns=cols,
                          index=[f"DS{i}" for i in range(20)])
    tex = avg_rank_tex("F1", higher_is_better=True, means=means)
    assert tex.count("\\texttt{") == 50
    # rank-1 combo (highest value, since higher_is_better) must be bolded first.
    assert r"\textbf{\texttt{FAM4.s9}}" in tex


if __name__ == "__main__":
    test_tie_splits_one_over_n()
    test_row_sums_equal_dataset_count()
    test_best_count_respects_direction()
    test_tie_rule_uses_four_decimal_rounding()
    test_fmt_value_switches_to_scientific()
    test_avg_rank_table_lists_all_50_combos_sorted()
    print("All Summary Tables tests passed.")
