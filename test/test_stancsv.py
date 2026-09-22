"""testing stancsv parsing"""

import io
import os
from pathlib import Path
from test import without_import

import numpy as np
import pytest

import cmdstanpy
from cmdstanpy.utils import stancsv

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


def test_csv_bytes_to_numpy() -> None:
    lines = [
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
        b"-6.81411,0.983499,0.787025,1,1,0,6.8147,0.20649\n",
        b"-6.85511,0.994945,0.787025,2,3,0,6.85536,0.310589\n",
        b"-6.85511,0.812189,0.787025,1,1,0,7.16517,0.310589\n",
    ]
    expected = np.array(
        [
            [-6.76206, 1, 0.787025, 1, 1, 0, 6.81411, 0.229458],
            [-6.81411, 0.983499, 0.787025, 1, 1, 0, 6.8147, 0.20649],
            [-6.85511, 0.994945, 0.787025, 2, 3, 0, 6.85536, 0.310589],
            [-6.85511, 0.812189, 0.787025, 1, 1, 0, 7.16517, 0.310589],
        ],
        dtype=np.float64,
    )
    arr_out = stancsv.csv_bytes_list_to_numpy(lines)
    assert np.array_equal(arr_out, expected)
    assert arr_out[0].dtype == np.float64


def test_csv_bytes_to_numpy_no_polars() -> None:
    lines = [
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
        b"-6.81411,0.983499,0.787025,1,1,0,6.8147,0.20649\n",
        b"-6.85511,0.994945,0.787025,2,3,0,6.85536,0.310589\n",
        b"-6.85511,0.812189,0.787025,1,1,0,7.16517,0.310589\n",
    ]
    expected = np.array(
        [
            [-6.76206, 1, 0.787025, 1, 1, 0, 6.81411, 0.229458],
            [-6.81411, 0.983499, 0.787025, 1, 1, 0, 6.8147, 0.20649],
            [-6.85511, 0.994945, 0.787025, 2, 3, 0, 6.85536, 0.310589],
            [-6.85511, 0.812189, 0.787025, 1, 1, 0, 7.16517, 0.310589],
        ],
        dtype=np.float64,
    )
    with without_import("polars", cmdstanpy.utils.stancsv):
        arr_out = stancsv.csv_bytes_list_to_numpy(lines)
        assert np.array_equal(arr_out, expected)
        assert arr_out[0].dtype == np.float64


def test_csv_bytes_to_numpy_single_element() -> None:
    lines = [
        b"-6.76206\n",
    ]
    expected = np.array(
        [
            [-6.76206],
        ],
        dtype=np.float64,
    )
    arr_out = stancsv.csv_bytes_list_to_numpy(lines)
    assert np.array_equal(arr_out, expected)


def test_csv_bytes_to_numpy_single_element_no_polars() -> None:
    lines = [
        b"-6.76206\n",
    ]
    expected = np.array(
        [
            [-6.76206],
        ],
        dtype=np.float64,
    )
    with without_import("polars", cmdstanpy.utils.stancsv):
        arr_out = stancsv.csv_bytes_list_to_numpy(lines)
        assert np.array_equal(arr_out, expected)


def test_csv_bytes_empty() -> None:
    arr = stancsv.csv_bytes_list_to_numpy([])
    assert np.array_equal(arr, np.empty((0, 0)))


def test_parse_comments_header_and_draws() -> None:
    lines: list[bytes] = [b"# 1\n", b"a\n", b"3\n", b"# 4\n"]
    (
        comment_lines,
        header,
        draws_lines,
    ) = stancsv.parse_comments_header_and_draws(iter(lines))

    assert comment_lines == [b"# 1\n", b"# 4\n"]
    assert header == "a"
    assert draws_lines == [b"3\n"]


def test_scan_header() -> None:
    lines: list[bytes] = [b"# 1\n", b"a,b\n", b"3,4\n", b"# 4\n"]
    assert stancsv.scan_header(iter(lines)) == "a,b"
    assert stancsv.scan_header(iter([b"# 1\n", b"# 2\n"])) is None

    csv_path = os.path.join(DATAFILES_PATH, "logistic_output_1.csv")
    header = stancsv.scan_header(csv_path)
    assert header is not None
    assert header.startswith("lp__")


def test_csv_polars_and_numpy_equiv() -> None:
    lines = [
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
        b"-6.81411,0.983499,0.787025,1,1,0,6.8147,0.20649\n",
        b"-6.85511,0.994945,0.787025,2,3,0,6.85536,0.310589\n",
        b"-6.85511,0.812189,0.787025,1,1,0,7.16517,0.310589\n",
    ]
    arr_out_polars = stancsv.csv_bytes_list_to_numpy(lines)
    with without_import("polars", cmdstanpy.utils.stancsv):
        arr_out_numpy = stancsv.csv_bytes_list_to_numpy(lines)
    assert np.array_equal(arr_out_polars, arr_out_numpy)


def test_csv_polars_and_numpy_equiv_one_line() -> None:
    lines = [
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
    ]
    arr_out_polars = stancsv.csv_bytes_list_to_numpy(lines)
    with without_import("polars", cmdstanpy.utils.stancsv):
        arr_out_numpy = stancsv.csv_bytes_list_to_numpy(lines)
    assert np.array_equal(arr_out_polars, arr_out_numpy)


def test_csv_polars_and_numpy_equiv_one_element() -> None:
    lines = [
        b"-6.76206\n",
    ]
    arr_out_polars = stancsv.csv_bytes_list_to_numpy(lines)
    with without_import("polars", cmdstanpy.utils.stancsv):
        arr_out_numpy = stancsv.csv_bytes_list_to_numpy(lines)
    assert np.array_equal(arr_out_polars, arr_out_numpy)


def test_parse_stan_csv_from_file() -> None:
    csv_path = os.path.join(DATAFILES_PATH, "bernoulli_output_1.csv")

    (
        comment_lines,
        header,
        draws_lines,
    ) = stancsv.parse_comments_header_and_draws(csv_path)
    assert all(ln.startswith(b"#") for ln in comment_lines)
    assert header is not None and not header.startswith("#")
    assert all(not ln.startswith(b"#") for ln in draws_lines)

    (
        comment_lines_path,
        header_path,
        draws_lines_path,
    ) = stancsv.parse_comments_header_and_draws(Path(csv_path))
    assert all(ln.startswith(b"#") for ln in comment_lines_path)
    assert header_path is not None and not header.startswith("#")
    assert all(not ln.startswith(b"#") for ln in draws_lines_path)

    assert comment_lines == comment_lines_path
    assert header == header_path
    assert draws_lines == draws_lines_path


def test_column_filter_basic() -> None:
    data = [b"1,2,3\n", b"4,5,6\n"]
    indexes = [0, 2]
    expected = [b"1,3\n", b"4,6\n"]
    assert stancsv.filter_csv_bytes_by_columns(data, indexes) == expected


def test_column_filter_empty_input() -> None:
    assert not stancsv.filter_csv_bytes_by_columns([], [0])


def test_column_filter_empty_indexes() -> None:
    data = [b"1,2,3\n", b"4,5,6\n"]
    assert stancsv.filter_csv_bytes_by_columns(data, []) == [b"\n", b"\n"]


def test_column_filter_single_column() -> None:
    data = [b"a,b,c\n", b"d,e,f\n"]
    assert stancsv.filter_csv_bytes_by_columns(data, [1]) == [b"b\n", b"e\n"]


def test_column_filter_non_consecutive_indexes() -> None:
    data = [b"9,8,7,6\n", b"5,4,3,2\n"]
    assert stancsv.filter_csv_bytes_by_columns(data, [2, 0]) == [
        b"7,9\n",
        b"3,5\n",
    ]


def test_parse_header() -> None:
    header = (
        "lp__,accept_stat__,stepsize__,treedepth__"
        ",n_leapfrog__,divergent__,energy__,theta.1"
    )
    parsed = stancsv.parse_header(header)
    expected = (
        "lp__",
        "accept_stat__",
        "stepsize__",
        "treedepth__",
        "n_leapfrog__",
        "divergent__",
        "energy__",
        "theta[1]",
    )
    assert parsed == expected


def test_parse_variational_eta() -> None:
    csv_path = os.path.join(DATAFILES_PATH, "variational", "eta_big_output.csv")
    comments, *_ = stancsv.parse_comments_header_and_draws(csv_path)
    eta = stancsv.parse_variational_eta(comments)
    assert eta == 100.0


def test_parse_variational_eta_no_block() -> None:
    comments = [
        b"# stanc_version = stanc3 v2.28.0\n",
        b"# stancflags = \n",
        b"lp__,log_p__,log_g__,mu.1,mu.2\n",
        b"0,0,0,311.545,532.801\n",
        b"0,-186118,-4.74553,311.545,353.503\n",
        b"0,-184982,-2.75303,311.545,587.377\n",
    ]

    with pytest.raises(ValueError):
        stancsv.parse_variational_eta(comments)


def test_max_treedepth_and_divergence_counts() -> None:
    header = (
        "lp__,accept_stat__,stepsize__,treedepth__,"
        "n_leapfrog__,divergent__,energy__,theta\n"
    )
    draws = [
        b"-4.78686,0.986298,1.09169,1,3,0,5.29492,0.550024\n",
        b"-5.07942,0.676947,1.09169,10,3,0,6.44279,0.709113\n",
        b"-5.04922,1,1.09169,1,1,0,5.14176,0.702445\n",
        b"-5.09338,0.996111,1.09169,10,3,1,5.16083,0.712059\n",
        b"-4.78903,0.989798,1.09169,1,3,0,5.08116,0.546685\n",
        b"-5.36502,0.854345,1.09169,1,3,0,5.39311,0.369686\n",
        b"-5.13605,0.937837,1.09169,1,3,0,5.95811,0.720607\n",
        b"-4.80646,1,1.09169,2,3,0,5.0962,0.528418\n",
    ]
    out = stancsv.extract_max_treedepth_and_divergence_counts(
        header, draws, 10, 0
    )
    assert out == (2, 1)


def test_max_treedepth_and_divergence_counts_warmup_draws() -> None:
    header = (
        "lp__,accept_stat__,stepsize__,treedepth__,"
        "n_leapfrog__,divergent__,energy__,theta\n"
    )
    draws = [
        b"-4.78686,0.986298,1.09169,1,3,0,5.29492,0.550024\n",
        b"-5.07942,0.676947,1.09169,10,3,0,6.44279,0.709113\n",
        b"-5.04922,1,1.09169,1,1,0,5.14176,0.702445\n",
        b"-5.09338,0.996111,1.09169,10,3,1,5.16083,0.712059\n",
        b"-4.78903,0.989798,1.09169,1,3,0,5.08116,0.546685\n",
        b"-5.36502,0.854345,1.09169,1,3,0,5.39311,0.369686\n",
        b"-5.13605,0.937837,1.09169,1,3,0,5.95811,0.720607\n",
        b"-4.80646,1,1.09169,2,3,0,5.0962,0.528418\n",
    ]
    out = stancsv.extract_max_treedepth_and_divergence_counts(
        header, draws, 10, 2
    )
    assert out == (1, 1)


def test_max_treedepth_and_divergence_counts_no_draws() -> None:
    header = (
        "lp__,accept_stat__,stepsize__,treedepth__,"
        "n_leapfrog__,divergent__,energy__,theta\n"
    )
    out = stancsv.extract_max_treedepth_and_divergence_counts(header, [], 10, 0)
    assert out == (0, 0)


def test_max_treedepth_and_divergence_invalid() -> None:
    header = "lp__,accept_stat__,stepsize__,n_leapfrog__,energy__,theta\n"
    draws = [
        b"-4.78686,0.986298,1.09169,3,5.29492,0.550024\n",
    ]
    assert stancsv.extract_max_treedepth_and_divergence_counts(
        header, draws, 10, 0
    ) == (0, 0)


def test_sneaky_fixed_param_check() -> None:
    sneaky_header = "lp__,accept_stat__,N,y_sim.1"
    normal_header = (
        "lp__,accept_stat__,stepsize__,treedepth__,"
        "n_leapfrog__,divergent__,energy__,theta"
    )

    assert stancsv.is_sneaky_fixed_param(sneaky_header)
    assert not stancsv.is_sneaky_fixed_param(normal_header)


def test_warmup_sampling_draw_counts() -> None:
    csv_path = os.path.join(DATAFILES_PATH, "bernoulli_output_1.csv")
    assert stancsv.count_warmup_and_sampling_draws(csv_path) == (0, 10)


def test_warmup_sampling_draw_counts_with_warmup() -> None:
    lines = [
        b"#     algorithm = hmc (Default)\n",
        (
            b"lp__,accept_stat__,stepsize__,treedepth__,"
            b"n_leapfrog__,divergent__,energy__,theta\n"
        ),
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
        b"# Adaptation terminated\n",
        b"# Step size = 0.787025\n",
        b"# Diagonal elements of inverse mass matrix:\n",
        b"# 1\n",
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
        b"# \n",
        b"#  Elapsed Time: 0.001332 seconds (Warm-up)\n",
    ]
    fio = io.BytesIO(b"".join(lines))
    assert stancsv.count_warmup_and_sampling_draws(fio) == (1, 1)


def test_warmup_sampling_draw_counts_fixed_param() -> None:
    lines = [
        b"#     algorithm = fixed_param\n",
        (
            b"lp__,accept_stat__,stepsize__,treedepth__,"
            b"n_leapfrog__,divergent__,energy__,theta\n"
        ),
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
        b"# \n",
        b"#  Elapsed Time: 0.001332 seconds (Warm-up)\n",
    ]
    fio = io.BytesIO(b"".join(lines))
    assert stancsv.count_warmup_and_sampling_draws(fio) == (0, 2)


def test_warmup_sampling_draw_counts_no_draws() -> None:
    lines = [
        b"#     algorithm = fixed_param\n",
        (
            b"lp__,accept_stat__,stepsize__,treedepth__,"
            b"n_leapfrog__,divergent__,energy__,theta\n"
        ),
        b"#  Elapsed Time: 0.001332 seconds (Warm-up)\n",
        b"#                0.001332 seconds (Sampling)\n",
    ]
    fio = io.BytesIO(b"".join(lines))
    assert stancsv.count_warmup_and_sampling_draws(fio) == (0, 0)


def test_warmup_sampling_draw_counts_invalid() -> None:
    lines = [
        b"#     algorithm = fixed_param\n",
    ]
    fio = io.BytesIO(b"".join(lines))
    with pytest.raises(ValueError):
        stancsv.count_warmup_and_sampling_draws(fio)


def test_inconsistent_draws_shape() -> None:
    header = "a,b"
    draws = [b"0,1,2\n"]
    with pytest.raises(ValueError):
        stancsv.raise_on_inconsistent_draws_shape(header, draws)


def test_inconsistent_draws_shape_empty() -> None:
    stancsv.raise_on_inconsistent_draws_shape("", [])


def test_munge_varname() -> None:
    name1 = "a"
    name2 = "a:1"
    name3 = "a:1.2"
    assert stancsv.munge_varname(name1) == "a"
    assert stancsv.munge_varname(name2) == "a.1"
    assert stancsv.munge_varname(name3) == "a.1[2]"
