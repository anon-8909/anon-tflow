import torch


def estimate_tail(
    data,
    output_file_path,
    number_of_bins,
    r_smooth,
    alpha,
    hsteps,
    bootstrap_flag,
    t_bootstrap,
    r_bootstrap,
    diagn_plots,
    eps_stop,
    theta1,
    theta2,
    verbose,
    noise_flag,
    p_noise,
    savedata,
):
    # convert to ascending
    ordered_data = data.sort().values

    log_x_pdf, y_pdf = compute_log_binned_pdf(
        ordered_data, number_of_bins=number_of_bins
    )

    x_ccdf, y_ccdf = compute_ccdf(ordered_data)


# utils
def compute_log_binned_pdf(ordered_data, number_of_bins=30):
    # define the support of the distribution
    lower_bound = torch.min(ordered_data)
    upper_bound = torch.max(ordered_data)

    # define bin edges
    if lower_bound <= 0:
        raise ValueError("Data must be strictly positive for log-binning.")

    # define bin edges
    # slightly widen to be sure top observations are included
    lower_bound_log = torch.log10(lower_bound) - 1e-5
    upper_bound_log = torch.log10(upper_bound) + 1e-5
    bins = torch.logspace(
        lower_bound_log,
        upper_bound_log,
        number_of_bins,
        device=ordered_data.device,
        dtype=ordered_data.dtype,
    )

    # compute the histogram using numpy
    histogram, bin_edges = torch.histogram(ordered_data, bins=bins)
    bin_widths = bins[1:] - bins[:-1]
    pdf_values = histogram / (torch.sum(histogram) * bin_widths)
    bin_midpoints = bin_edges[:-1] + bin_widths / 2

    # if bin is empty, drop it from the resulting list
    non_zero_indices = pdf_values > 0
    bin_midpoints = bin_midpoints[non_zero_indices]
    pdf_values = pdf_values[non_zero_indices]

    return bin_midpoints, pdf_values


def compute_ccdf(ordered_data):
    n = ordered_data.shape[0]
    ccdf = (
        1.0
        - torch.arange(
            1,
            n + 1,
            device=ordered_data.device,
            dtype=ordered_data.dtype,
        )
        / n
    )
    # flip to give descending order (common CCDF convention)
    return ordered_data.flip(0), ccdf.flip(0)
