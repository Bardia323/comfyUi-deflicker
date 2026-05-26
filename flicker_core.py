"""Temporal flicker removal for video frame sequences.

Uses the industry-standard timelapse deflicker approach:
1. Compute per-frame luminance/statistics.
2. Temporally smooth the statistics curve.
3. Apply gain/offset correction to match each frame to the smooth target.
4. Multi-pass iteration for convergence (like LRTimelapse visual deflicker).

Enhanced with adaptive trend detection to balance flicker removal
vs. preserving intentional brightness changes.
"""
import torch
import torch.nn.functional as F


def _safe_empty_cache():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Border masking — exclude black bars from statistics
# ---------------------------------------------------------------------------

def _compute_content_mask(images: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
    """Detect content vs. black border regions for stabilized/cropped footage.

    Computes a spatial mask [H, W] where True = content pixel, False = border.
    A pixel is considered border if its mean brightness across ALL frames is
    below the threshold. This catches letterbox, pillarbox, and irregular
    stabilization crops.

    The mask is only used for statistics computation (frame means, CDFs).
    Corrections are always applied to the full frame.

    Args:
        images: [B, H, W, C] sRGB tensor in [0, 1].
        threshold: Brightness below this (across all frames) = border. Default
            0.02 catches near-black borders without masking dark content.

    Returns:
        [H, W] boolean mask. True = content pixel.
    """
    # Mean brightness per pixel across all frames and channels
    # Compute incrementally in chunks to avoid memory spikes on massive tensors
    B, H, W, C = images.shape
    chunk_size = 32
    running_sum = torch.zeros(H, W, device=images.device, dtype=torch.float64)
    for i in range(0, B, chunk_size):
        chunk = images[i:i+chunk_size]
        running_sum += chunk.mean(dim=-1).sum(dim=0)
    temporal_mean = (running_sum / B).to(images.dtype)
    mask = temporal_mean >= threshold

    # Safety: if mask excludes >95% of pixels, it's probably a very dark
    # scene, not actual borders — fall back to using everything.
    if mask.sum() < mask.numel() * 0.05:
        return torch.ones_like(mask, dtype=torch.bool)

    return mask


# ---------------------------------------------------------------------------
# Temporal smoothing utilities
# ---------------------------------------------------------------------------

def _temporal_gaussian_kernel(window_size: int) -> torch.Tensor:
    """Create a 1D Gaussian kernel for temporal smoothing."""
    sigma = window_size / 4.0
    half = window_size // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / max(sigma, 0.5)) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def temporal_smooth(values: torch.Tensor, window_size: int) -> torch.Tensor:
    """Gaussian temporal smoothing of a 1D signal [num_frames].

    Uses reflect padding to avoid edge artifacts.
    """
    if window_size <= 1 or len(values) <= 1:
        return values.clone()

    window_size = min(window_size, len(values))
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return values.clone()

    kernel = _temporal_gaussian_kernel(window_size).to(values.device)
    half = len(kernel) // 2

    padded = F.pad(
        values.unsqueeze(0).unsqueeze(0),
        (half, half),
        mode="reflect",
    )
    smoothed = F.conv1d(padded, kernel.unsqueeze(0).unsqueeze(0))
    return smoothed.squeeze(0).squeeze(0)


def temporal_median_smooth(values: torch.Tensor, window_size: int) -> torch.Tensor:
    """Median temporal smoothing — robust to outliers."""
    if window_size <= 1 or len(values) <= 1:
        return values.clone()

    n = len(values)
    half = window_size // 2
    result = torch.empty_like(values)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = values[lo:hi].median()

    return temporal_smooth(result, max(3, window_size // 3))


# ---------------------------------------------------------------------------
# Step removal — instant correction of latent space shifts
# ---------------------------------------------------------------------------

def _remove_steps(
    ch_data: torch.Tensor,
    content_mask: torch.Tensor | None = None,
    threshold_mult: float = 5.0,
    strength: float = 1.0,
) -> torch.Tensor:
    """Remove step discontinuities (latent space shifts) from a channel.

    Detects sharp frame-to-frame jumps and applies affine correction
    (gain + contrast/gamma) to anchor all frames to the stable reference
    level measured before the first detected step.

    strength <= 1.0: blend between original and fully corrected.
    strength > 1.0: increases detection sensitivity (catches smaller steps)
    without causing progressive brightness drift.

    Algorithm:
    1. Compute per-frame means AND stds (masked).
    2. Compute frame-to-frame diffs for both stats.
    3. Detect outlier mean diffs (> threshold / detection_boost).
    4. Accumulate detected steps for mean and std as running corrections.
    5. Apply gain (mean) + contrast scaling (gamma) per frame.

    Args:
        ch_data: [N, H, W] single channel data.
        content_mask: [H, W] bool mask. True = content pixel for stats.
        threshold_mult: Sensitivity — how many times the median abs diff
            counts as a step. Lower = more sensitive. Default 5.0.
        strength: 0.0 = no correction, 1.0 = full step removal,
            >1.0 = more sensitive detection (no drift).

    Returns:
        Corrected [N, H, W] tensor (unclamped — caller handles clamping).
    """
    N = ch_data.shape[0]
    if N < 3:
        return ch_data.clone()

    # Compute per-frame stats incrementally to avoid allocating massive stats_data copies
    frame_means = torch.empty(N, dtype=ch_data.dtype, device=ch_data.device)
    frame_stds = torch.empty(N, dtype=ch_data.dtype, device=ch_data.device)
    if content_mask is not None:
        mask_flat = content_mask.reshape(-1)
        has_mask = mask_flat.any()
    else:
        has_mask = False

    for i in range(N):
        frame = ch_data[i]
        if has_mask:
            frame_flat = frame.reshape(-1)[mask_flat]
        else:
            frame_flat = frame.reshape(-1)
        frame_means[i] = frame_flat.mean()
        frame_stds[i] = frame_flat.std()

    # Frame-to-frame diffs
    mean_diffs = frame_means[1:] - frame_means[:-1]
    std_diffs = frame_stds[1:] - frame_stds[:-1]

    # Threshold: outlier diffs relative to the typical noise.
    median_abs_diff = mean_diffs.abs().median()
    floor = frame_means.mean().item() * 0.005  # 0.5% of mean brightness
    threshold = max(median_abs_diff.item() * threshold_mult, floor)

    if threshold < 1e-6:
        return ch_data.clone()

    # strength > 1.0 → lower threshold (catch smaller steps)
    # correction always at 100% to prevent progressive drift
    detection_boost = max(strength, 1.0)
    effective_threshold = threshold / detection_boost
    correction_blend = min(strength, 1.0)

    # Detect steps
    is_step = mean_diffs.abs() > effective_threshold

    if not is_step.any():
        return ch_data.clone()

    # Accumulate step corrections for both mean and std (vectorized)
    mean_corrections = torch.where(is_step, -mean_diffs, torch.zeros_like(mean_diffs))
    std_corrections = torch.where(is_step, -std_diffs, torch.zeros_like(std_diffs))

    cum_mean = torch.zeros(N, device=ch_data.device)
    cum_std = torch.zeros(N, device=ch_data.device)
    cum_mean[1:] = mean_corrections.cumsum(dim=0)
    cum_std[1:] = std_corrections.cumsum(dim=0)

    # Blend for strength < 1.0 (partial correction)
    cum_mean = cum_mean * correction_blend
    cum_std = cum_std * correction_blend

    # Target stats (anchored to first stable segment)
    target_means = frame_means + cum_mean
    target_stds = (frame_stds + cum_std).clamp(min=1e-4)
    safe_means = frame_means.clamp(min=1e-2)
    safe_stds = frame_stds.clamp(min=1e-4)

    # Step 1: Gain correction for mean (preserves black = 0)
    gains = (target_means / safe_means).clamp(0.25, 4.0)

    # Step 2: Contrast/gamma correction via std matching
    # After gain, effective std = frame_std * gain
    effective_stds = (safe_stds * gains).clamp(min=1e-4)
    contrast_ratios = (target_stds / effective_stds).clamp(0.5, 2.0)

    # Apply: gain first, then contrast around the new mean
    gains_3d = gains.view(-1, 1, 1)
    target_means_3d = target_means.view(-1, 1, 1)
    contrast_3d = contrast_ratios.view(-1, 1, 1)

    gained = ch_data * gains_3d
    corrected = target_means_3d + (gained - target_means_3d) * contrast_3d

    return corrected


# ---------------------------------------------------------------------------
# Adaptive trend detection
# ---------------------------------------------------------------------------

def _masked_frame_means(
    images: torch.Tensor, content_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Compute per-frame mean brightness using only content pixels.

    Args:
        images: [B, H, W, C] or [B, H, W].
        content_mask: [H, W] bool mask, or None for all pixels.

    Returns:
        [B] tensor of per-frame means.
    """
    B = images.shape[0]
    means = torch.empty(B, dtype=images.dtype, device=images.device)

    if content_mask is None:
        chunk_size = 32
        for i in range(0, B, chunk_size):
            chunk = images[i:i+chunk_size]
            if images.dim() == 4:
                means[i:i+chunk_size] = chunk.mean(dim=(1, 2, 3))
            else:
                means[i:i+chunk_size] = chunk.mean(dim=(1, 2))
        return means

    # Compute frame-by-frame to avoid allocating massive intermediate flat copies
    if images.dim() == 4:
        for i in range(B):
            frame = images[i]
            means[i] = frame[content_mask].mean()
    else:
        for i in range(B):
            frame = images[i]
            means[i] = frame[content_mask].mean()
    return means


def _detect_trend(frame_means: torch.Tensor) -> bool:
    """Detect whether there's a significant linear trend in the frame means.

    Returns True if the trend magnitude significantly exceeds the noise level,
    indicating intentional brightness changes that should be preserved.
    """
    N = len(frame_means)
    if N < 5:
        return False

    t = torch.arange(N, dtype=torch.float32, device=frame_means.device)
    t_centered = t - t.mean()
    m_centered = frame_means - frame_means.mean()

    # Linear regression slope
    slope = (m_centered * t_centered).sum() / (t_centered ** 2).sum()

    # Trend magnitude: total brightness change across the sequence
    trend_magnitude = abs(slope.item()) * N

    # Compare against flicker magnitude (std of means)
    flicker_magnitude = frame_means.std().item()

    # Trend is "significant" if total change > 2x the noise level
    return trend_magnitude > flicker_magnitude * 2.0


# ---------------------------------------------------------------------------
# Core correction: global_mean + wide_trend + iteration
# ---------------------------------------------------------------------------

def _correct_channel(
    ch_data: torch.Tensor,
    window_size: int,
    strength: float,
    smooth_fn,
    has_trend: bool = False,
    content_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Correct a single channel's temporal flicker using gain-based correction.

    Uses multiplicative (gain) correction instead of affine (mean+std) to
    preserve black levels: pixel * gain keeps 0 at 0, like an exposure dial.

    Algorithm:
    - No trend: flatten means to global constant via gain.
    - Trend detected: fit polynomial or wide-window smooth, apply via gain.

    Args:
        ch_data: [N, H, W] single channel.
        window_size: user's temporal smoothing window.
        strength: correction strength (0-1).
        smooth_fn: temporal smoothing function.
        has_trend: if True, preserve slow brightness changes.
        content_mask: [H, W] bool mask. True = content pixel for stats.

    Returns:
        Corrected [N, H, W] (unclamped — caller handles clamping).
    """
    num_frames = ch_data.shape[0]
    # Compute per-frame means incrementally to avoid allocating massive intermediate copies
    frame_means = torch.empty(num_frames, dtype=ch_data.dtype, device=ch_data.device)
    if content_mask is not None:
        mask_flat = content_mask.reshape(-1)
        has_mask = mask_flat.any()
    else:
        has_mask = False

    for i in range(num_frames):
        frame = ch_data[i]
        if has_mask:
            frame_means[i] = frame.reshape(-1)[mask_flat].mean()
        else:
            frame_means[i] = frame.mean()
    global_mean = frame_means.mean()

    if has_trend:
        # Trend detected: fit a low-order polynomial to capture the smooth
        # underlying brightness curve — like a compositor drawing a spline
        # in Nuke. The polynomial gives a perfectly smooth target that
        # removes ALL per-frame noise while preserving the overall shape.
        #
        # If the polynomial fits poorly (e.g. step function at a chunk
        # boundary), fall back to iterative wide-window smoothing.
        degree = max(2, min(5, num_frames // 20))
        t = torch.arange(num_frames, dtype=torch.float32, device=ch_data.device)
        t_norm = t / max(num_frames - 1, 1)
        V = torch.stack([t_norm ** d for d in range(degree + 1)], dim=1)
        coeffs = torch.linalg.solve(V.T @ V, V.T @ frame_means)
        poly_target = V @ coeffs

        # Check fit quality: if max residual > 3x the typical noise,
        # the polynomial doesn't capture the signal well (step function).
        poly_residual = (frame_means - poly_target).abs()
        noise_estimate = frame_means.std() * 0.5
        if poly_residual.max() > noise_estimate * 3:
            # Poor polynomial fit — use iterative wide-window smoothing.
            wide_w = min(window_size * 2, num_frames)
            if wide_w % 2 == 0:
                wide_w = max(3, wide_w - 1)
            corrected_means = frame_means.clone()
            for _ in range(3):
                trend = smooth_fn(corrected_means, wide_w)
                tc = trend - trend.mean()
                corrected_means = global_mean + tc
            target_means = corrected_means
        else:
            target_means = poly_target
    else:
        # No significant trend: flatten to constant → near-100% removal.
        target_means = torch.full_like(frame_means, global_mean)

    # Blend with strength
    final_means = frame_means + (target_means - frame_means) * strength

    # --- Apply gain-based correction per frame ---
    # gain = target / current — keeps 0 at 0 (like exposure compensation)
    safe_means = frame_means.clamp(min=1e-2)
    gains = (final_means / safe_means).clamp(0.25, 4.0).view(-1, 1, 1)

    corrected = ch_data * gains

    return corrected


def _correct_channel_grid(
    ch_data: torch.Tensor,
    window_size: int,
    strength: float,
    smooth_fn,
    has_trend: bool,
    grid_size: int,
    content_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Correct a single channel using per-cell gain on a spatial grid.

    Computes per-cell means, then applies multiplicative gain correction
    per cell with bilinear-interpolated seams.

    Args:
        ch_data: [N, H, W] single channel.
        window_size, strength, smooth_fn, has_trend: same as _correct_channel.
        grid_size: number of cells per axis.
        content_mask: [H, W] bool mask. True = content pixel for stats.

    Returns:
        Corrected [N, H, W].
    """
    N, H, W = ch_data.shape
    device = ch_data.device

    # Fast path: no mask → use adaptive_avg_pool2d (fused GPU kernel)
    if content_mask is None:
        data_4d = ch_data.unsqueeze(1)
        cell_means_pooled = F.adaptive_avg_pool2d(data_4d, grid_size).squeeze(1)  # [N, G, G]

    cell_h = H / grid_size
    cell_w = W / grid_size

    # For each cell: compute gain = target_mean / current_mean
    gains = torch.ones(N, grid_size, grid_size, device=device)

    for gy in range(grid_size):
        for gx in range(grid_size):
            if content_mask is None:
                # Fast path: use pooled means
                cm = cell_means_pooled[:, gy, gx]  # [N]
            else:
                y0 = round(gy * cell_h)
                y1 = round((gy + 1) * cell_h)
                x0 = round(gx * cell_w)
                x1 = round((gx + 1) * cell_w)

                cell_data = ch_data[:, y0:y1, x0:x1]
                cell_flat = cell_data.reshape(N, -1)
                cell_mask = content_mask[y0:y1, x0:x1].reshape(-1)
                if cell_mask.any():
                    cm = cell_flat[:, cell_mask].mean(dim=1)
                else:
                    # Entire cell is border — skip correction
                    continue

            global_mean = cm.mean()

            if has_trend:
                target_means = smooth_fn(cm, window_size)
            else:
                target_means = torch.full_like(cm, global_mean)

            final_means = cm + (target_means - cm) * strength
            safe_means = cm.clamp(min=1e-2)

            gains[:, gy, gx] = (final_means / safe_means).clamp(0.25, 4.0)

    # Upsample gain map from [N, G, G] to [N, H, W] with bilinear interpolation
    gains_up = F.interpolate(
        gains.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False,
    ).squeeze(1)  # [N, H, W]

    return ch_data * gains_up


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Target working-set size for one band of the per-pixel temporal pass. The
# whole video is never reshaped at once — we slice the frame into row-bands and
# run the conv over each band, so peak RAM stays near this budget regardless of
# clip length. Bump this if you have RAM to spare and want fewer, larger bands.
_PIXEL_BAND_BUDGET_BYTES = 512 * 1024 * 1024


def _pixel_temporal_smooth(
    images: torch.Tensor,
    window_size: int,
    blend_strength: float,
    inplace: bool = False,
) -> torch.Tensor:
    """Per-pixel temporal smoothing (inspired by SuperBeasts PixelDeflicker).

    For each pixel, runs a Gaussian-weighted average across neighbouring frames
    within a sliding window — a genuine 1D convolution over the time axis (via
    ``F.conv1d``), not an approximation. This removes spatially-varying flicker
    that frame-mean correction cannot address. The result is blended with the
    input using ``blend_strength`` to preserve detail.

    Memory: the whole video is never materialised at once. The frame is sliced
    into horizontal row-bands and the convolution runs band-by-band, so peak
    RAM stays near ``_PIXEL_BAND_BUDGET_BYTES`` no matter how many frames are
    passed in. Each pixel's temporal convolution is independent of every other
    pixel, so banding is exact — bit-for-bit identical to convolving the whole
    tensor in one shot.

    Args:
        images: [B, H, W, C] tensor.
        window_size: temporal window for averaging.
        blend_strength: 0=no pixel smoothing, 1=full replacement.
        inplace: if True, write the blended result back into ``images`` instead
            of allocating a new output buffer. Safe because each band is fully
            read (and copied) before its slice is overwritten. Only pass True
            when the caller owns ``images``.

    Returns:
        Blended [B, H, W, C] tensor (``images`` itself when ``inplace``).
    """
    if blend_strength <= 0 or window_size <= 1:
        return images

    num_frames = images.shape[0]
    window_size = min(window_size, num_frames)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return images

    try:
        import comfy.model_management
        device = comfy.model_management.get_torch_device()
    except ImportError:
        device = images.device

    # Gaussian weights for the temporal window
    sigma = window_size / 4.0
    half = window_size // 2
    t = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    weights = torch.exp(-0.5 * (t / max(sigma, 0.5)) ** 2)
    weights = weights / weights.sum()  # [K]
    kernel = weights.flip(0).view(1, 1, -1).to(device)  # [1, 1, K]

    B, H, W, C = images.shape
    result = images if inplace else torch.empty_like(images)

    # Band height chosen so one band's conv intermediates stay near the budget.
    # F.conv1d unfolds the kernel internally, so its transient workspace scales
    # with the window size (~K), not just a fixed number of copies. Size the
    # band by (window_size + a few copies) so a wide window doesn't blow past
    # the budget. Always at least one row, so even huge frames make progress.
    bytes_per_row = W * C * B * images.element_size()
    per_row_workspace = bytes_per_row * (window_size + 4)
    rows = max(1, int(_PIXEL_BAND_BUDGET_BYTES / max(per_row_workspace, 1)))

    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        # band is a view; permute+reshape forces a contiguous copy, so the
        # conv reads a snapshot and writing back to images[:, y0:y1] is safe.
        band = images[:, y0:y1]                                  # [B, h, W, C]
        
        # Copy ONLY this band to the GPU
        band_device = band.to(device)
        flat = band_device.permute(1, 2, 3, 0).reshape(-1, 1, B)        # [h*W*C, 1, B]
        padded = F.pad(flat, (half, half), mode="reflect")       # reflect = no edge bias
        smoothed = F.conv1d(padded, kernel)                      # [h*W*C, 1, B]
        smoothed = smoothed.reshape(y1 - y0, W, C, B).permute(3, 0, 1, 2)
        
        # Blend: original * (1-strength) + smoothed * strength
        blended = band_device * (1.0 - blend_strength) + smoothed * blend_strength
        
        # Copy back to result on its original device
        result[:, y0:y1] = blended.to(result.device)
        
        # Free memory immediately
        del band_device, flat, padded, smoothed, blended
        _safe_empty_cache()

    return result


@torch.no_grad()
def deflicker_frames(
    images: torch.Tensor,
    window_size: int = 15,
    strength: float = 1.0,
    channels: str = "L",
    use_median: bool = False,
    pixel_smoothing: float = 0.0,
    grid_size: int = 1,
    drift_mode: str = "auto",
    content_mask: torch.Tensor | None = None,
    mode: str = "temporal_smoothing",
    gen_heatmap: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove temporal brightness/color flicker from a frame sequence.

    Supports two correction modes:
    - temporal_smoothing: Gaussian/median smoothing of per-frame statistics.
      Best for random per-frame flicker (AI video noise).
    - step_removal: Instant correction of sharp brightness steps caused by
      latent space shifts. Preserves natural trends, removes only discontinuities.
    - both: Step removal first, then temporal smoothing.

    Args:
        images: [B, H, W, 3] sRGB tensor in [0, 1].
        window_size: Temporal smoothing window in frames. Controls the
            separation between "flicker" (removed) and "trend" (preserved).
            Only used in temporal_smoothing/both modes.
        strength: 0.0 = no correction, 1.0 = full correction.
        channels: "L" = brightness only (uniform delta across RGB),
                  "LAB" = per-channel correction (fixes color flicker too).
        use_median: Use median pre-filter (robust to extreme outlier frames).
        pixel_smoothing: Per-pixel temporal smoothing strength (0=off, 1=full).
        grid_size: Spatial grid for correction. 1 = global, >1 = per-cell.
        drift_mode: "auto" = detect trend automatically,
                    "flicker_only" = remove all changes including slow drift,
                    "preserve_trend" = always keep slow brightness changes.
        content_mask: [H, W] bool mask from _compute_content_mask(). If None,
            computed automatically.
        mode: "temporal_smoothing" = classic window-based correction,
              "step_removal" = instant step discontinuity correction,
              "both" = step removal then temporal smoothing.
        gen_heatmap: If True, build the debug heatmap (an extra full-size
            [B, H, W, 3] buffer). Set False on long sequences / tight RAM to
            skip that allocation; a tiny placeholder is returned instead.

    Returns:
        (corrected_images, debug_heatmap). corrected is [B, H, W, 3]; the
        heatmap is [B, H, W, 3] when gen_heatmap is True, else a [1, 1, 1, 3]
        placeholder.
    """
    num_frames, H, W, C = images.shape
    device = images.device

    if num_frames < 2 or strength <= 0:
        hm_shape = (num_frames, H, W, 3) if gen_heatmap else (1, 1, 1, 3)
        return images, torch.zeros(*hm_shape, device=device)

    smooth_fn = temporal_median_smooth if use_median else temporal_smooth

    # Auto-detect black borders (stabilized/cropped footage)
    if content_mask is None:
        content_mask = _compute_content_mask(images)

    do_steps = mode in ("step_removal", "both")
    do_temporal = mode in ("temporal_smoothing", "both")

    corrected = images
    # We mutate in place by default to completely avoid holding two full-size
    # [N, H, W, 3] copies of the video in host RAM.
    owns_buffer = True

    def _apply_gain(buf, gain_map, owns):
        """Multiply the full frame buffer by a broadcast gain map, in place
        when we own the buffer. Returns (new_buffer, owns=True)."""
        gain_map = gain_map.clamp(0.25, 4.0)
        if owns:
            buf.mul_(gain_map).clamp_(0.0, 1.0)
            return buf, True
        return (buf * gain_map).clamp(0.0, 1.0), True

    # --- Phase 0: Step removal (latent space shift correction) ---
    if do_steps:
        if channels == "L":
            brightness = corrected.mean(dim=-1)  # [N, H, W]
            corrected_brightness = _remove_steps(
                brightness, content_mask, strength=strength,
            )
            gain_map = (corrected_brightness / brightness.clamp(min=1e-4)).unsqueeze(-1)
            del brightness, corrected_brightness
            corrected, owns_buffer = _apply_gain(corrected, gain_map, owns_buffer)
            del gain_map
        else:
            for ch in range(3):
                corrected[..., ch] = _remove_steps(
                    corrected[..., ch], content_mask, strength=strength,
                ).clamp(0.0, 1.0)
        
        # Aggressive garbage collection
        _safe_empty_cache()

    # --- Phase 1: Per-frame statistics correction (temporal smoothing) ---
    if do_temporal:
        # Determine trend handling based on drift_mode
        if drift_mode == "flicker_only":
            has_trend = False
        elif drift_mode == "preserve_trend":
            has_trend = True
        else:
            brightness_means = _masked_frame_means(corrected, content_mask)
            has_trend = _detect_trend(brightness_means)

        correct_fn = (
            lambda ch, ws, st, sf, ht: _correct_channel(ch, ws, st, sf, ht, content_mask)
        ) if grid_size <= 1 else (
            lambda ch, ws, st, sf, ht: _correct_channel_grid(
                ch, ws, st, sf, ht, grid_size, content_mask,
            )
        )

        if channels == "L":
            brightness = corrected.mean(dim=-1)
            corrected_brightness = correct_fn(
                brightness, window_size, strength, smooth_fn, has_trend,
            )
            gain_map = (corrected_brightness / brightness.clamp(min=1e-4)).unsqueeze(-1)
            del brightness, corrected_brightness
            corrected, owns_buffer = _apply_gain(corrected, gain_map, owns_buffer)
            del gain_map
        else:
            for ch in range(3):
                corrected[..., ch] = correct_fn(
                    corrected[..., ch], window_size, strength, smooth_fn, has_trend,
                ).clamp(0.0, 1.0)
        
        # Aggressive garbage collection
        _safe_empty_cache()

    # --- Phase 2: Per-pixel temporal smoothing (optional) ---
    if pixel_smoothing > 0 and do_temporal:
        # By Phase 2 we always own `corrected` (Phase 1 produced it), so the
        # banded pass can write back in place — no second full-video buffer.
        corrected = _pixel_temporal_smooth(
            corrected, window_size, pixel_smoothing * strength,
            inplace=owns_buffer,
        )
        corrected.clamp_(0.0, 1.0)
        
        # Aggressive garbage collection
        _safe_empty_cache()

    if gen_heatmap:
        heatmap = _generate_correction_heatmap(corrected, images)
    else:
        heatmap = torch.zeros(1, 1, 1, 3, device=device)
    return corrected, heatmap


def _generate_correction_heatmap(
    corrected: torch.Tensor, original: torch.Tensor,
) -> torch.Tensor:
    """Blue-black-red heatmap from brightness difference."""
    B, H, W, C = corrected.shape
    device = corrected.device
    
    # Compute max_abs incrementally in chunks to avoid memory spikes
    chunk_size = 32
    max_abs = 0.0
    for i in range(0, B, chunk_size):
        c_chunk = corrected[i:i+chunk_size]
        o_chunk = original[i:i+chunk_size]
        diff_chunk = c_chunk.mean(dim=-1) - o_chunk.mean(dim=-1)
        max_abs = max(max_abs, diff_chunk.abs().max().item())
        del diff_chunk
        _safe_empty_cache()

    heatmap = torch.zeros(B, H, W, 3, dtype=corrected.dtype, device=device)
    if max_abs < 1e-6:
        return heatmap

    # Populate heatmap incrementally
    for i in range(0, B, chunk_size):
        c_chunk = corrected[i:i+chunk_size]
        o_chunk = original[i:i+chunk_size]
        diff_chunk = c_chunk.mean(dim=-1) - o_chunk.mean(dim=-1)
        normalized = diff_chunk / max_abs
        
        heatmap[i:i+chunk_size, ..., 0] = normalized.clamp(min=0)
        heatmap[i:i+chunk_size, ..., 2] = (-normalized).clamp(min=0)
        
        del diff_chunk, normalized
        _safe_empty_cache()

    return heatmap
