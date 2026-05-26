import torch

from .brightness_core import auto_brightness_equalize
from .flicker_core import deflicker_frames, _compute_content_mask, _generate_correction_heatmap


class DeflickerFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["step_removal", "both", "temporal_smoothing"], {
                    "default": "step_removal",
                    "tooltip": "Step removal: instant correction of sharp latent space shifts. Temporal smoothing: Gaussian window-based correction for random flicker. Both: step removal first, then temporal smoothing.",
                }),
                "channels": (["L", "LAB"], {
                    "default": "L",
                    "tooltip": "L: brightness only — preserves original colors. LAB: brightness + color correction.",
                }),
                "step_strength": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Step detection sensitivity. 1 = full correction, >1 = catches smaller steps without drift. Ignored in temporal_smoothing mode.",
                }),
                "smooth_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Temporal smoothing strength. 1 = full, >1 = overcorrect. Ignored in step_removal mode.",
                }),
                "smooth_window": ("INT", {
                    "default": 25, "min": 3, "max": 999, "step": 2,
                    "tooltip": "Temporal smoothing window (frames). Larger = more aggressive. Ignored in step_removal mode.",
                }),
                "smooth_drift": (["auto", "flicker_only", "preserve_trend"], {
                    "default": "auto",
                    "tooltip": "Ignored in step_removal mode. Auto: detect trend automatically. Flicker only: remove all brightness changes. Preserve trend: keep slow changes.",
                }),
                "smooth_median": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Ignored in step_removal mode. Median pre-filter for extreme outlier frames.",
                }),
                "smooth_pixel": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Ignored in step_removal mode. Per-pixel temporal smoothing. 0=off, 0.3-0.5=AI video.",
                }),
                "smooth_grid": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Spatial grid for correction. 1 = global, 6 = 6x6 zones. Used by temporal smoothing and equalize.",
                }),
                "eq_enable": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto brightness equalize: detect and smooth chunk boundary jumps after deflicker.",
                }),
                "eq_blend_radius": ("INT", {
                    "default": 5, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Equalize: max frames to blend around each boundary.",
                }),
                "eq_sensitivity": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 6.0, "step": 0.5,
                    "tooltip": "Equalize: detection sensitivity. Lower = more sensitive.",
                }),
                "output_heatmap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate the debug_heatmap output. Turn OFF on long clips / low RAM — it saves a full-size image buffer. When off, debug_heatmap is a 1x1 placeholder.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "debug_heatmap")
    FUNCTION = "deflicker"
    CATEGORY = "deflicker"

    def deflicker(self, images, mode, channels, step_strength, smooth_strength,
                  smooth_window, smooth_drift, smooth_median, smooth_pixel,
                  smooth_grid, eq_enable, eq_blend_radius, eq_sensitivity,
                  output_heatmap=True):
        # Compute content mask once from original images (excludes black borders)
        content_mask = _compute_content_mask(images)

        if mode == "both":
            # Run step removal and temporal smoothing with separate strengths.
            # The intermediate pass never needs a heatmap.
            corrected, _ = deflicker_frames(
                images=images, window_size=smooth_window,
                strength=step_strength,
                channels=channels, use_median=smooth_median,
                pixel_smoothing=smooth_pixel, grid_size=smooth_grid,
                drift_mode=smooth_drift, content_mask=content_mask,
                mode="step_removal", gen_heatmap=False,
            )
            corrected, _ = deflicker_frames(
                images=corrected, window_size=smooth_window,
                strength=smooth_strength,
                channels=channels, use_median=smooth_median,
                pixel_smoothing=smooth_pixel, grid_size=smooth_grid,
                drift_mode=smooth_drift, content_mask=content_mask,
                mode="temporal_smoothing", gen_heatmap=False,
            )
            # Heatmap shows total correction vs original
            if output_heatmap:
                heatmap = _generate_correction_heatmap(corrected, images)
            else:
                heatmap = torch.zeros(1, 1, 1, 3, device=images.device)
        else:
            strength = step_strength if mode == "step_removal" else smooth_strength
            corrected, heatmap = deflicker_frames(
                images=images, window_size=smooth_window,
                strength=strength,
                channels=channels, use_median=smooth_median,
                pixel_smoothing=smooth_pixel, grid_size=smooth_grid,
                drift_mode=smooth_drift, content_mask=content_mask,
                mode=mode, gen_heatmap=output_heatmap,
            )

        # Equalize (boundary smoothing)
        if eq_enable:
            corrected, eq_heatmap = auto_brightness_equalize(
                images=corrected, blend_radius=eq_blend_radius,
                strength=1.0, sensitivity=eq_sensitivity,
                grid_size=smooth_grid, content_mask=content_mask,
            )

        return (corrected, heatmap)
