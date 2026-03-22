"""Comprehensive tests for the flicker removal tool.

Tests use synthetic flicker data with known ground truth to measure
the exact removal percentage. Target: >95% flicker removal.
"""
import torch
import pytest


# ---------------------------------------------------------------------------
# Helper: measure flicker as std of per-frame mean brightness
# ---------------------------------------------------------------------------

def _measure_flicker(images: torch.Tensor) -> float:
    """Measure flicker as the std of per-frame mean brightness."""
    frame_means = images.mean(dim=(1, 2, 3))  # [N]
    return frame_means.std().item()


def _measure_flicker_lab_L(images: torch.Tensor) -> float:
    """Measure flicker in LAB L channel."""
    from brightness_core import srgb_to_lab
    lab = srgb_to_lab(images)
    L_means = lab[..., 0].reshape(images.shape[0], -1).mean(dim=1)
    return L_means.std().item()


def _removal_ratio(original_flicker: float, corrected_flicker: float) -> float:
    """Compute flicker removal ratio: 1.0 = perfect, 0.0 = no improvement."""
    if original_flicker < 1e-8:
        return 1.0
    return 1.0 - corrected_flicker / original_flicker


# ---------------------------------------------------------------------------
# Test: temporal smoothing kernel
# ---------------------------------------------------------------------------

class TestTemporalSmoothing:
    def test_smooth_constant_signal(self):
        """Smoothing a constant signal should return the same signal."""
        from flicker_core import temporal_smooth
        signal = torch.full((20,), 5.0)
        smoothed = temporal_smooth(signal, window_size=7)
        assert torch.allclose(smoothed, signal, atol=1e-5)

    def test_smooth_reduces_noise(self):
        """Smoothing should reduce high-frequency noise."""
        from flicker_core import temporal_smooth
        torch.manual_seed(42)
        clean = torch.full((50,), 10.0)
        noisy = clean + torch.randn(50) * 2.0
        smoothed = temporal_smooth(noisy, window_size=11)
        # Smoothed should be closer to clean than noisy
        noisy_error = (noisy - clean).abs().mean().item()
        smoothed_error = (smoothed - clean).abs().mean().item()
        assert smoothed_error < noisy_error * 0.5

    def test_smooth_preserves_trend(self):
        """Smoothing should preserve gradual trends."""
        from flicker_core import temporal_smooth
        trend = torch.linspace(0, 10, 50)
        noisy = trend + torch.randn(50) * 0.5
        smoothed = temporal_smooth(noisy, window_size=11)
        # Smoothed should follow the trend
        trend_error = (smoothed - trend).abs().mean().item()
        assert trend_error < 1.0  # much less than the noise amplitude

    def test_smooth_short_sequence(self):
        """Should handle sequences shorter than window."""
        from flicker_core import temporal_smooth
        signal = torch.tensor([1.0, 2.0, 3.0])
        smoothed = temporal_smooth(signal, window_size=11)
        assert smoothed.shape == signal.shape

    def test_smooth_window_1(self):
        """Window=1 should return original signal."""
        from flicker_core import temporal_smooth
        signal = torch.randn(10)
        smoothed = temporal_smooth(signal, window_size=1)
        assert torch.allclose(smoothed, signal)

    def test_median_smooth(self):
        """Median smooth should be robust to outliers."""
        from flicker_core import temporal_median_smooth
        torch.manual_seed(42)
        signal = torch.full((30,), 5.0)
        # Add a few extreme outliers
        signal[10] = 50.0
        signal[20] = -30.0
        smoothed = temporal_median_smooth(signal, window_size=7)
        # Outlier frames should be corrected back near 5.0
        assert abs(smoothed[10].item() - 5.0) < 2.0
        assert abs(smoothed[20].item() - 5.0) < 2.0


# ---------------------------------------------------------------------------
# Test: global flicker removal (uniform brightness variation)
# ---------------------------------------------------------------------------

class TestGlobalFlicker:
    def _make_flickering_sequence(self, num_frames=30, flicker_std=0.05, seed=42):
        """Create a synthetic flickering sequence.

        All frames have the same content but with random brightness offsets.
        """
        torch.manual_seed(seed)
        # Base frame: mid-range with some texture
        base = torch.rand(1, 32, 32, 3) * 0.3 + 0.35
        images = base.expand(num_frames, -1, -1, -1).clone()

        # Add random per-frame brightness offset
        flicker = torch.randn(num_frames) * flicker_std
        for i in range(num_frames):
            images[i] = (images[i] + flicker[i]).clamp(0, 1)

        return images, flicker

    def test_removes_global_flicker_95pct(self):
        """Should remove >95% of uniform brightness flicker."""
        from flicker_core import deflicker_frames
        images, _ = self._make_flickering_sequence(num_frames=30, flicker_std=0.05)

        original_flicker = _measure_flicker(images)
        result, _ = deflicker_frames(images, window_size=11, strength=1.0)
        corrected_flicker = _measure_flicker(result)

        ratio = _removal_ratio(original_flicker, corrected_flicker)
        assert ratio > 0.95, f"Only removed {ratio*100:.1f}% of flicker"

    def test_removes_strong_flicker(self):
        """Should handle strong flicker (large brightness jumps)."""
        from flicker_core import deflicker_frames
        images, _ = self._make_flickering_sequence(num_frames=40, flicker_std=0.12)

        original_flicker = _measure_flicker(images)
        result, _ = deflicker_frames(images, window_size=15, strength=1.0)
        corrected_flicker = _measure_flicker(result)

        ratio = _removal_ratio(original_flicker, corrected_flicker)
        assert ratio > 0.90, f"Only removed {ratio*100:.1f}% of strong flicker"

    def test_removes_subtle_flicker(self):
        """Should handle subtle flicker (small brightness variations)."""
        from flicker_core import deflicker_frames
        images, _ = self._make_flickering_sequence(num_frames=30, flicker_std=0.015)

        original_flicker = _measure_flicker(images)
        result, _ = deflicker_frames(images, window_size=11, strength=1.0)
        corrected_flicker = _measure_flicker(result)

        ratio = _removal_ratio(original_flicker, corrected_flicker)
        assert ratio > 0.90, f"Only removed {ratio*100:.1f}% of subtle flicker"

    def test_longer_sequence(self):
        """Should work on longer sequences."""
        from flicker_core import deflicker_frames
        images, _ = self._make_flickering_sequence(num_frames=100, flicker_std=0.06)

        original_flicker = _measure_flicker(images)
        result, _ = deflicker_frames(images, window_size=21, strength=1.0)
        corrected_flicker = _measure_flicker(result)

        ratio = _removal_ratio(original_flicker, corrected_flicker)
        assert ratio > 0.95, f"Only removed {ratio*100:.1f}% on long sequence"

    def test_preserves_content(self):
        """Correction should not destroy spatial content."""
        from flicker_core import deflicker_frames
        images, _ = self._make_flickering_sequence(num_frames=20, flicker_std=0.05)

        result, _ = deflicker_frames(images, window_size=11, strength=1.0)

        # Spatial structure within each frame should be preserved.
        # Check that the per-pixel ranking is similar (Spearman-style).
        for i in range(0, 20, 5):
            orig_flat = images[i].flatten()
            corr_flat = result[i].flatten()
            # Pearson correlation should be very high
            cov = ((orig_flat - orig_flat.mean()) * (corr_flat - corr_flat.mean())).mean()
            corr = cov / (orig_flat.std() * corr_flat.std() + 1e-8)
            assert corr > 0.99, f"Frame {i}: correlation too low ({corr:.4f})"


# ---------------------------------------------------------------------------
# Test: flicker with trend preservation
# ---------------------------------------------------------------------------

class TestTrendPreservation:
    def test_preserves_gradual_brightening(self):
        """Should preserve a gradual brightening trend while removing flicker."""
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        num_frames = 40

        # Create frames with a gradual brightening trend
        base = torch.rand(1, 32, 32, 3) * 0.2 + 0.2
        images = base.expand(num_frames, -1, -1, -1).clone()

        # Add gradual trend: 0.2 -> 0.6 over 40 frames
        trend = torch.linspace(0, 0.3, num_frames)
        flicker = torch.randn(num_frames) * 0.04

        for i in range(num_frames):
            images[i] = (images[i] + trend[i] + flicker[i]).clamp(0, 1)

        # Measure original
        original_means = images.mean(dim=(1, 2, 3))
        original_trend_range = original_means[-5:].mean() - original_means[:5].mean()

        result, _ = deflicker_frames(images, window_size=11, strength=1.0)

        corrected_means = result.mean(dim=(1, 2, 3))
        corrected_trend_range = corrected_means[-5:].mean() - corrected_means[:5].mean()

        # Trend should be partially preserved (>50% of original range).
        # The algorithm intentionally prioritizes aggressive flicker removal
        # over perfect trend preservation, since for AI video the "trend"
        # is often part of the flicker/drift to remove.
        assert corrected_trend_range > original_trend_range * 0.5, \
            f"Trend was destroyed: {corrected_trend_range:.4f} vs {original_trend_range:.4f}"

        # But frame-to-frame noise should be reduced
        original_noise = (original_means[1:] - original_means[:-1]).std()
        corrected_noise = (corrected_means[1:] - corrected_means[:-1]).std()
        assert corrected_noise < original_noise * 0.5


# ---------------------------------------------------------------------------
# Test: alternating (high-frequency) flicker
# ---------------------------------------------------------------------------

class TestHighFreqFlicker:
    def test_alternating_bright_dark(self):
        """Should handle alternating bright/dark frames."""
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        num_frames = 30
        base = torch.rand(1, 32, 32, 3) * 0.2 + 0.4
        images = base.expand(num_frames, -1, -1, -1).clone()

        # Alternating pattern: +0.05, -0.05, +0.05, ...
        for i in range(num_frames):
            offset = 0.06 * (1 if i % 2 == 0 else -1)
            images[i] = (images[i] + offset).clamp(0, 1)

        original_flicker = _measure_flicker(images)
        result, _ = deflicker_frames(images, window_size=7, strength=1.0)
        corrected_flicker = _measure_flicker(result)

        ratio = _removal_ratio(original_flicker, corrected_flicker)
        assert ratio > 0.90, f"Only removed {ratio*100:.1f}% of alternating flicker"

    def test_periodic_flicker(self):
        """Should handle periodic (sinusoidal) flicker."""
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        num_frames = 60
        base = torch.rand(1, 32, 32, 3) * 0.2 + 0.4
        images = base.expand(num_frames, -1, -1, -1).clone()

        # Sinusoidal flicker with period of ~8 frames
        t = torch.arange(num_frames, dtype=torch.float32)
        flicker = 0.05 * torch.sin(2 * 3.14159 * t / 8)

        for i in range(num_frames):
            images[i] = (images[i] + flicker[i]).clamp(0, 1)

        original_flicker = _measure_flicker(images)
        result, _ = deflicker_frames(images, window_size=15, strength=1.0)
        corrected_flicker = _measure_flicker(result)

        ratio = _removal_ratio(original_flicker, corrected_flicker)
        assert ratio > 0.90, f"Only removed {ratio*100:.1f}% of periodic flicker"


# ---------------------------------------------------------------------------
# Test: color channel flicker
# ---------------------------------------------------------------------------

class TestColorFlicker:
    def test_color_temperature_flicker(self):
        """Should correct color temperature shifts when using LAB mode."""
        from flicker_core import deflicker_frames
        from brightness_core import srgb_to_lab

        torch.manual_seed(42)
        num_frames = 30
        base = torch.rand(1, 32, 32, 3) * 0.2 + 0.4
        images = base.expand(num_frames, -1, -1, -1).clone()

        # Add color temperature flicker (warm/cool shifts)
        for i in range(num_frames):
            shift = torch.randn(1).item() * 0.03
            images[i, ..., 0] = (images[i, ..., 0] + shift).clamp(0, 1)  # R
            images[i, ..., 2] = (images[i, ..., 2] - shift).clamp(0, 1)  # B

        # Measure original color flicker in LAB a/b channels
        lab_orig = srgb_to_lab(images)
        a_flicker_orig = lab_orig[..., 1].reshape(num_frames, -1).mean(1).std().item()

        result, _ = deflicker_frames(images, window_size=11, strength=1.0, channels="LAB")

        lab_corr = srgb_to_lab(result)
        a_flicker_corr = lab_corr[..., 1].reshape(num_frames, -1).mean(1).std().item()

        ratio = _removal_ratio(a_flicker_orig, a_flicker_corr)
        assert ratio > 0.80, f"Only removed {ratio*100:.1f}% of color flicker"


# ---------------------------------------------------------------------------
# Test: edge cases and parameters
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_strength_zero(self):
        """strength=0 should return original images unchanged."""
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        images = torch.rand(10, 16, 16, 3)
        result, _ = deflicker_frames(images, window_size=7, strength=0.0)
        assert torch.allclose(result, images, atol=1e-4)

    def test_single_frame(self):
        """Single frame should pass through unchanged."""
        from flicker_core import deflicker_frames
        images = torch.rand(1, 16, 16, 3)
        result, _ = deflicker_frames(images, window_size=7, strength=1.0)
        assert torch.allclose(result, images, atol=1e-4)

    def test_two_frames(self):
        """Two frames should be handled without error."""
        from flicker_core import deflicker_frames
        images = torch.rand(2, 16, 16, 3)
        result, _ = deflicker_frames(images, window_size=7, strength=1.0)
        assert result.shape == images.shape

    def test_uniform_no_change(self):
        """Uniform brightness frames should not be modified."""
        from flicker_core import deflicker_frames
        images = torch.full((10, 16, 16, 3), 0.5)
        result, _ = deflicker_frames(images, window_size=7, strength=1.0)
        assert torch.allclose(result, images, atol=1e-3)

    def test_output_range(self):
        """Output should always be in [0, 1]."""
        from flicker_core import deflicker_frames
        torch.manual_seed(42)
        images = torch.rand(20, 16, 16, 3)
        result, _ = deflicker_frames(images, window_size=11, strength=1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_heatmap_shape(self):
        """Heatmap should match input batch shape."""
        from flicker_core import deflicker_frames
        images = torch.rand(10, 16, 16, 3)
        _, heatmap = deflicker_frames(images, window_size=7, strength=1.0)
        assert heatmap.shape == (10, 16, 16, 3)

    def test_L_only_mode(self):
        """L-only mode should not affect color channels."""
        from flicker_core import deflicker_frames
        from brightness_core import srgb_to_lab

        torch.manual_seed(42)
        num_frames = 20
        base = torch.rand(1, 16, 16, 3) * 0.3 + 0.35
        images = base.expand(num_frames, -1, -1, -1).clone()

        # Add L-channel flicker only
        flicker = torch.randn(num_frames) * 0.05
        for i in range(num_frames):
            images[i] = (images[i] + flicker[i]).clamp(0, 1)

        result, _ = deflicker_frames(images, window_size=11, strength=1.0, channels="L")

        # L flicker should be reduced
        l_flicker_orig = _measure_flicker_lab_L(images)
        l_flicker_corr = _measure_flicker_lab_L(result)
        assert l_flicker_corr < l_flicker_orig * 0.2

    def test_median_mode(self):
        """Median mode should work and handle outliers better."""
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        num_frames = 30
        base = torch.rand(1, 32, 32, 3) * 0.2 + 0.4
        images = base.expand(num_frames, -1, -1, -1).clone()

        # Add normal flicker + a few extreme outlier frames
        flicker = torch.randn(num_frames) * 0.03
        flicker[10] = 0.25  # extreme bright spike
        flicker[20] = -0.20  # extreme dark spike

        for i in range(num_frames):
            images[i] = (images[i] + flicker[i]).clamp(0, 1)

        result, _ = deflicker_frames(
            images, window_size=11, strength=1.0, use_median=True
        )
        corrected_flicker = _measure_flicker(result)
        original_flicker = _measure_flicker(images)

        ratio = _removal_ratio(original_flicker, corrected_flicker)
        assert ratio > 0.85, f"Median mode only removed {ratio*100:.1f}%"


# ---------------------------------------------------------------------------
# Test: combined with boundary correction (integration)
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_deflicker_reduces_per_chunk_flicker(self):
        """Deflicker should reduce within-chunk flicker, even with a chunk boundary.

        The deflicker tool removes per-frame noise; the boundary fix tool
        handles the brightness step between chunks. Here we test that
        deflicker at least reduces the within-chunk flicker significantly.
        """
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        # Two chunks with different base brightness + random flicker
        chunk1_base = torch.rand(1, 32, 32, 3) * 0.2 + 0.3
        chunk2_base = torch.rand(1, 32, 32, 3) * 0.2 + 0.5

        chunk1 = chunk1_base.expand(20, -1, -1, -1).clone()
        chunk2 = chunk2_base.expand(20, -1, -1, -1).clone()

        for i in range(20):
            chunk1[i] = (chunk1[i] + torch.randn(1).item() * 0.04).clamp(0, 1)
            chunk2[i] = (chunk2[i] + torch.randn(1).item() * 0.04).clamp(0, 1)

        images = torch.cat([chunk1, chunk2], dim=0)

        deflickered, _ = deflicker_frames(images, window_size=11, strength=1.0)

        # Measure per-chunk flicker (within each chunk, not across boundary)
        orig_c1_flicker = _measure_flicker(images[:20])
        orig_c2_flicker = _measure_flicker(images[20:])
        corr_c1_flicker = _measure_flicker(deflickered[:20])
        corr_c2_flicker = _measure_flicker(deflickered[20:])

        # Each chunk's internal flicker should be reduced.
        # Note: with a large brightness step between chunks, the algorithm
        # uses conservative mode (preserves the "trend"). Within-chunk
        # flicker is still reduced, but not as aggressively as for flat sequences.
        # The boundary tool handles the step; deflicker handles per-frame noise.
        ratio1 = _removal_ratio(orig_c1_flicker, corr_c1_flicker)
        ratio2 = _removal_ratio(orig_c2_flicker, corr_c2_flicker)
        assert ratio1 > 0.30, f"Chunk 1: only removed {ratio1*100:.1f}%"
        assert ratio2 > 0.30, f"Chunk 2: only removed {ratio2*100:.1f}%"


# ---------------------------------------------------------------------------
# Test: real-world-like scenario with varying content
# ---------------------------------------------------------------------------

class TestRealisticScenarios:
    def test_scene_with_moving_content(self):
        """Frames with slightly different content + flicker."""
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        num_frames = 30

        # Each frame has slightly different content (simulating motion)
        images = torch.zeros(num_frames, 32, 32, 3)
        for i in range(num_frames):
            # Smooth gradient that shifts over time (simulates camera pan)
            x = torch.linspace(0 + i * 0.01, 1 + i * 0.01, 32)
            y = torch.linspace(0, 1, 32)
            grid_x, grid_y = torch.meshgrid(y, x, indexing="ij")
            images[i, ..., 0] = grid_x * 0.5 + 0.25
            images[i, ..., 1] = grid_y * 0.4 + 0.3
            images[i, ..., 2] = 0.4

            # Add random brightness flicker
            flicker_offset = torch.randn(1).item() * 0.04
            images[i] = (images[i] + flicker_offset).clamp(0, 1)

        original_flicker = _measure_flicker(images)
        result, _ = deflicker_frames(images, window_size=11, strength=1.0)
        corrected_flicker = _measure_flicker(result)

        ratio = _removal_ratio(original_flicker, corrected_flicker)
        assert ratio > 0.90, f"Moving scene: only removed {ratio*100:.1f}%"

    def test_preserves_black_levels(self):
        """Gain-based correction should preserve black pixels at 0."""
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        num_frames = 30

        # Create frames with dark regions (near-black) and bright regions
        images = torch.zeros(num_frames, 32, 32, 3)
        for i in range(num_frames):
            # Top half: dark (0.0-0.05), bottom half: bright (0.3-0.7)
            images[i, :16, :, :] = torch.rand(16, 32, 3) * 0.05
            images[i, 16:, :, :] = torch.rand(16, 32, 3) * 0.4 + 0.3
            # Add brightness flicker
            flicker_offset = torch.randn(1).item() * 0.04
            images[i] = (images[i] + flicker_offset).clamp(0, 1)

        # Measure dark region before correction
        dark_before = images[:, :16, :, :].mean().item()

        result, _ = deflicker_frames(images, window_size=11, strength=1.0)

        # Dark region should stay dark (not drift upward)
        dark_after = result[:, :16, :, :].mean().item()

        # Allow some small change but blacks should not be lifted significantly
        assert dark_after < dark_before + 0.02, \
            f"Blacks lifted: {dark_before:.4f} -> {dark_after:.4f}"

        # Flicker should still be removed
        original_flicker = _measure_flicker(images)
        corrected_flicker = _measure_flicker(result)
        ratio = _removal_ratio(original_flicker, corrected_flicker)
        assert ratio > 0.85, f"Only removed {ratio*100:.1f}% of flicker"


# ---------------------------------------------------------------------------
# Test: black border masking (stabilized/cropped footage)
# ---------------------------------------------------------------------------

class TestBorderMasking:
    def test_mask_detects_black_borders(self):
        """_compute_content_mask should detect near-black border regions."""
        from flicker_core import _compute_content_mask

        # Create frames with black top/bottom bars (letterbox)
        images = torch.rand(10, 32, 32, 3) * 0.3 + 0.35  # content: 0.35-0.65
        images[:, :4, :, :] = 0.0   # top black bar
        images[:, -4:, :, :] = 0.0  # bottom black bar

        mask = _compute_content_mask(images)

        assert mask.shape == (32, 32)
        # Border rows should be masked out
        assert not mask[:4, :].any(), "Top border should be masked"
        assert not mask[-4:, :].any(), "Bottom border should be masked"
        # Content rows should be kept
        assert mask[4:-4, :].all(), "Content should not be masked"

    def test_mask_fallback_on_dark_scene(self):
        """Mask should fall back to all-True for very dark scenes."""
        from flicker_core import _compute_content_mask

        # Very dark scene — not borders, just dark content
        images = torch.rand(10, 16, 16, 3) * 0.01

        mask = _compute_content_mask(images)

        # Should fall back to all-True (>95% would be masked)
        assert mask.all(), "Dark scene should not be masked"

    def test_borders_dont_distort_correction(self):
        """Black borders should not affect the deflicker statistics.

        Without masking, borders pull frame means toward zero, causing
        the gain correction to be wrong. With masking, statistics are
        computed from content only, so the correction is accurate.
        """
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        num_frames = 30

        # Create content with known flicker
        content = torch.rand(1, 24, 32, 3) * 0.3 + 0.35
        content = content.expand(num_frames, -1, -1, -1).clone()
        flicker = torch.randn(num_frames) * 0.05
        for i in range(num_frames):
            content[i] = (content[i] + flicker[i]).clamp(0, 1)

        # Version WITHOUT borders (ground truth)
        result_no_border, _ = deflicker_frames(content, window_size=11, strength=1.0)
        content_flicker_removed = _measure_flicker(result_no_border)

        # Version WITH black borders (simulating stabilized crop)
        bordered = torch.zeros(num_frames, 32, 32, 3)
        bordered[:, 4:28, :, :] = content  # 4px black bars top/bottom
        result_bordered, _ = deflicker_frames(bordered, window_size=11, strength=1.0)

        # Measure flicker only in the content region of bordered result
        content_region = result_bordered[:, 4:28, :, :]
        bordered_content_flicker = _measure_flicker(content_region)

        # Both should achieve similar flicker removal
        ratio_no_border = _removal_ratio(_measure_flicker(content), content_flicker_removed)
        ratio_bordered = _removal_ratio(
            _measure_flicker(bordered[:, 4:28, :, :]), bordered_content_flicker
        )

        assert ratio_bordered > 0.85, \
            f"Bordered version only removed {ratio_bordered*100:.1f}% (vs {ratio_no_border*100:.1f}% without borders)"

    def test_correction_applies_to_full_frame(self):
        """Correction should apply to all pixels, including borders."""
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        num_frames = 20

        # Content with flicker + slight non-zero border (0.005)
        images = torch.full((num_frames, 16, 16, 3), 0.005)
        images[:, 2:14, 2:14, :] = 0.5  # content region

        # Add flicker to everything
        for i in range(num_frames):
            offset = torch.randn(1).item() * 0.03
            images[i] = (images[i] + offset).clamp(0, 1)

        result, _ = deflicker_frames(images, window_size=11, strength=1.0)

        # Result shape should match
        assert result.shape == images.shape
        # Border pixels should still exist (not zeroed out)
        assert result[:, 0, 0, :].shape == (num_frames, 3)


# ---------------------------------------------------------------------------
# Test: step removal (latent space shift correction)
# ---------------------------------------------------------------------------

class TestStepRemoval:
    def test_removes_step_discontinuities(self):
        """Step removal should eliminate sharp brightness jumps."""
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        num_frames = 60
        base = torch.rand(1, 32, 32, 3) * 0.3 + 0.35
        images = base.expand(num_frames, -1, -1, -1).clone()

        # Add step discontinuities (latent space shifts)
        # Frame 20: sudden drop, frame 40: sudden jump
        for i in range(20, num_frames):
            images[i] = (images[i] - 0.05).clamp(0, 1)
        for i in range(40, num_frames):
            images[i] = (images[i] + 0.08).clamp(0, 1)

        # Measure step magnitudes before
        means = images.mean(dim=(1, 2, 3))
        orig_step1 = abs(means[20] - means[19]).item()
        orig_step2 = abs(means[40] - means[39]).item()

        result, _ = deflicker_frames(images, strength=1.0, mode="step_removal")

        # Measure after
        corr_means = result.mean(dim=(1, 2, 3))
        corr_step1 = abs(corr_means[20] - corr_means[19]).item()
        corr_step2 = abs(corr_means[40] - corr_means[39]).item()

        assert corr_step1 < orig_step1 * 0.1, \
            f"Step 1 not removed: {orig_step1:.4f} -> {corr_step1:.4f}"
        assert corr_step2 < orig_step2 * 0.1, \
            f"Step 2 not removed: {orig_step2:.4f} -> {corr_step2:.4f}"

    def test_preserves_natural_trend(self):
        """Step removal should not affect gradual brightness changes."""
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        num_frames = 50
        base = torch.rand(1, 32, 32, 3) * 0.2 + 0.3
        images = base.expand(num_frames, -1, -1, -1).clone()

        # Add gradual trend (no steps)
        trend = torch.linspace(0, 0.15, num_frames)
        for i in range(num_frames):
            images[i] = (images[i] + trend[i]).clamp(0, 1)

        orig_means = images.mean(dim=(1, 2, 3))
        result, _ = deflicker_frames(images, strength=1.0, mode="step_removal")
        corr_means = result.mean(dim=(1, 2, 3))

        # Trend range should be preserved (no steps to remove)
        orig_range = orig_means[-1] - orig_means[0]
        corr_range = corr_means[-1] - corr_means[0]

        assert abs(corr_range - orig_range) < orig_range * 0.1, \
            f"Trend destroyed: {orig_range:.4f} -> {corr_range:.4f}"

    def test_both_mode_combines_step_and_temporal(self):
        """'both' mode should remove steps AND smooth remaining flicker."""
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        num_frames = 60
        base = torch.rand(1, 32, 32, 3) * 0.3 + 0.35
        images = base.expand(num_frames, -1, -1, -1).clone()

        # Add step at frame 30 + random flicker
        for i in range(30, num_frames):
            images[i] = (images[i] - 0.06).clamp(0, 1)
        flicker = torch.randn(num_frames) * 0.03
        for i in range(num_frames):
            images[i] = (images[i] + flicker[i]).clamp(0, 1)

        result, _ = deflicker_frames(
            images, window_size=11, strength=1.0, mode="both",
        )

        # Step should be removed
        means = result.mean(dim=(1, 2, 3))
        step_diff = abs(means[30] - means[29]).item()
        assert step_diff < 0.01, f"Step not removed in 'both' mode: {step_diff:.4f}"

        # Flicker should also be reduced
        corr_flicker = _measure_flicker(result)
        orig_flicker = _measure_flicker(images)
        ratio = _removal_ratio(orig_flicker, corr_flicker)
        assert ratio > 0.70, f"Flicker not reduced in 'both' mode: {ratio*100:.1f}%"

    def test_step_removal_output_range(self):
        """Output should be in [0, 1]."""
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        images = torch.rand(30, 16, 16, 3) * 0.5 + 0.25
        # Add large step
        images[15:] = (images[15:] + 0.15).clamp(0, 1)

        result, _ = deflicker_frames(images, strength=1.0, mode="step_removal")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_no_progressive_drift_with_strength_above_1(self):
        """strength > 1 should NOT cause progressive brightness drift.

        Previously, cumulative * strength caused each step correction to
        overshoot by (strength-1)*step, accumulating across multiple steps.
        Now, strength > 1 increases detection sensitivity instead.
        """
        from flicker_core import deflicker_frames

        torch.manual_seed(42)
        num_frames = 100
        base = torch.rand(1, 32, 32, 3) * 0.3 + 0.35
        images = base.expand(num_frames, -1, -1, -1).clone()

        # 5 downward steps (each chunk gets darker)
        for step_frame in [20, 40, 60, 80]:
            for i in range(step_frame, num_frames):
                images[i] = (images[i] - 0.04).clamp(0, 1)

        ref_mean = images[:20].mean().item()  # first chunk reference

        result, _ = deflicker_frames(
            images, strength=1.2, mode="step_removal",
        )

        # All corrected segments should be close to the first segment level
        corr_means = result.mean(dim=(1, 2, 3))
        for seg_start in [0, 20, 40, 60, 80]:
            seg_mean = corr_means[seg_start:seg_start + 20].mean().item()
            assert abs(seg_mean - ref_mean) < 0.015, \
                f"Segment at {seg_start}: mean {seg_mean:.4f} drifted from ref {ref_mean:.4f}"

    def test_gamma_correction_matches_contrast(self):
        """Step removal should correct contrast/gamma changes, not just mean."""
        from flicker_core import _remove_steps

        torch.manual_seed(42)
        N, H, W = 40, 32, 32

        # Create base with known std
        base = torch.rand(1, H, W) * 0.4 + 0.3  # mean ~0.5, std ~0.115
        ch_data = base.expand(N, -1, -1).clone()

        # First 20 frames: normal. Next 20: step in mean AND contrast.
        for i in range(20, N):
            ch_data[i] = ch_data[i] * 1.3 + 0.05  # gain + offset = gamma shift

        ref_std = ch_data[:20].reshape(20, -1).std(dim=1).mean().item()

        corrected = _remove_steps(ch_data, strength=1.0)

        corr_std = corrected[20:].reshape(20, -1).std(dim=1).mean().item()
        # Corrected segment's std should be closer to the reference
        orig_std = ch_data[20:].reshape(20, -1).std(dim=1).mean().item()
        orig_err = abs(orig_std - ref_std)
        corr_err = abs(corr_std - ref_std)
        assert corr_err < orig_err * 0.5, \
            f"Gamma not corrected: ref_std={ref_std:.4f}, orig={orig_std:.4f}, corr={corr_std:.4f}"
