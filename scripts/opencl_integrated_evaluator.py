"""OpenCL implementation of the prepared integrated-mass evaluator."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


OPENCL_KERNEL_SOURCE = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline double interp_linear(
    const double x,
    __global const double *xp,
    __global const double *fp,
    const int count
) {
    if (x <= xp[0]) {
        return fp[0];
    }
    if (x >= xp[count - 1]) {
        return fp[count - 1];
    }

    int high = 1;
    while (high < count && x > xp[high]) {
        ++high;
    }
    const int low = high - 1;
    const double denominator = xp[high] - xp[low];
    if (denominator == 0.0) {
        return fp[high];
    }
    const double fraction = (x - xp[low]) / denominator;
    return fp[low] + fraction * (fp[high] - fp[low]);
}

inline double clamp_unit(const double value) {
    return fmin(1.0, fmax(0.0, value));
}

__kernel void phase_mass(
    __global const double *signals,
    __global const double *weight_g,
    __global const double *weight_aq,
    __global const uchar *gas_constraint,
    __global const uchar *aqueous_constraint,
    __global const uchar *gas_score,
    __global const double *supports,
    __global const double *values,
    const int support_count,
    __global const double *lut_y,
    __global const double *lut_caq,
    const int lut_count,
    const double min_aq,
    const double max_aq,
    const double min_g,
    const double max_g,
    const float residual_onset,
    const float residual_width,
    const float arrival_scale,
    const float score_scale,
    const int residual_mode,
    const ulong pixel_count,
    __global double *mass_g,
    __global double *mass_aq
) {
    const ulong index = get_global_id(0);
    if (index >= pixel_count) {
        return;
    }

    const double signal = interp_linear(
        signals[index], supports, values, support_count
    );
    const double aq_denominator = max_aq == min_aq ? 1.0 : max_aq - min_aq;
    const double gas_denominator = max_g == min_g ? 1.0 : max_g - min_g;
    const double y_norm = clamp_unit((signal - min_aq) / aq_denominator);
    double concentration_aq = interp_linear(
        y_norm, lut_y, lut_caq, lut_count
    );
    concentration_aq *= aqueous_constraint[index] != 0;

    double saturation_g;
    if (residual_mode != 0) {
        const float concentration_f = (float)concentration_aq;
        const float score = (float)gas_score[index] * score_scale;
        float saturation_f = clamp(
            (score - residual_onset) / residual_width, 0.0f, 1.0f
        );
        saturation_f *= clamp(
            concentration_f / arrival_scale, 0.0f, 1.0f
        );
        concentration_aq = (double)concentration_f;
        saturation_g = (double)saturation_f;
    } else {
        saturation_g = clamp_unit(
            (fmin(max_g, fmax(min_g, signal)) - min_g) / gas_denominator
        );
        saturation_g *= gas_constraint[index] != 0;
    }

    mass_g[index] = weight_g[index] * saturation_g;
    mass_aq[index] = (
        weight_aq[index]
        * concentration_aq
        * fmax(0.0, 1.0 - saturation_g)
    );
}
"""


def _normalise_phase_separation(value: Any) -> str:
    mode = str(value or "shared-signal").strip().lower().replace("_", "-")
    if mode in {"residual", "residual-gas"}:
        return "residual-gas"
    if mode in {"shared", "shared-signal"}:
        return "shared-signal"
    raise ValueError(f"Unknown phase separation mode {value!r}")


def _select_device(cl: Any) -> tuple[Any, Any]:
    platform_filter = os.environ.get("FFAC_OPENCL_PLATFORM", "").strip().lower()
    device_filter = os.environ.get("FFAC_OPENCL_DEVICE", "").strip().lower()
    candidates: list[tuple[Any, Any]] = []
    for platform in cl.get_platforms():
        if platform_filter and platform_filter not in platform.name.lower():
            continue
        for device in platform.get_devices():
            if not device.type & cl.device_type.GPU:
                continue
            identity = " ".join(
                (platform.name, platform.vendor, device.name, device.vendor)
            ).lower()
            if device_filter and device_filter not in identity:
                continue
            candidates.append((platform, device))

    if not candidates:
        filters = f"platform={platform_filter!r}, device={device_filter!r}"
        raise RuntimeError(f"No matching OpenCL GPU was found ({filters})")
    for platform, device in candidates:
        identity = f"{platform.vendor} {device.vendor} {device.name}".lower()
        if "amd" in identity or "radeon" in identity:
            return platform, device
    return candidates[0]


class OpenCLIntegratedEvaluator:
    """Evaluate integrated masses while prepared frames remain in GPU memory."""

    def __init__(self, context: Any, *, residual_score_clip: float) -> None:
        try:
            import pyopencl as cl
            import pyopencl.array as cl_array
            from pyopencl.tools import ImmediateAllocator, MemoryPool
        except ImportError as exc:
            raise RuntimeError(
                "OpenCL evaluation requires the optional 'opencl' dependency"
            ) from exc
        if not context._prepared_colors:
            raise RuntimeError("OpenCL evaluation requires prepared color frames")

        calibration = context.calibration
        flash = calibration.flash
        if getattr(flash, "restoration", None) is not None:
            raise RuntimeError("OpenCL evaluation does not support flash restoration")

        platform, device = _select_device(cl)
        if "cl_khr_fp64" not in device.extensions.split():
            raise RuntimeError(
                f"OpenCL device {device.name!r} does not provide cl_khr_fp64"
            )

        self.cl = cl
        self.cl_array = cl_array
        self.context = context
        self.flash = flash
        self.phase_separation = _normalise_phase_separation(
            getattr(context, "phase_separation", "shared-signal")
        )
        self.arrival_scale = max(
            1e-6,
            float(os.environ.get("FFAC_RESIDUAL_GAS_ARRIVAL_SCALE", "0.02")),
        )
        self.score_scale = float(residual_score_clip) / 255.0
        self.cl_context = cl.Context([device])
        self.queue = cl.CommandQueue(self.cl_context)
        self.allocator = MemoryPool(ImmediateAllocator(self.queue))
        self.program = cl.Program(self.cl_context, OPENCL_KERNEL_SOURCE).build()
        self.phase_kernel = cl.Kernel(self.program, "phase_mass")
        self.platform = platform
        self.device = device

        shape = tuple(np.asarray(context._prepared_colors[0].img).shape[:2])
        context.geometry._prepare_cached_voxel_volume(list(shape))
        voxel_volume = np.asarray(context.geometry.cached_voxel_volume)
        density = np.asarray(calibration.co2_mass_analysis.density_gaseous_co2)
        solubility = np.asarray(calibration.co2_mass_analysis.solubility_co2)
        try:
            voxel_volume = np.broadcast_to(voxel_volume, shape)
            density = np.broadcast_to(density, shape)
            solubility = np.broadcast_to(solubility, shape)
        except ValueError as exc:
            raise RuntimeError(
                f"OpenCL static-array shape mismatch for {context.run}: "
                f"shape={shape}, voxel={voxel_volume.shape}, density={density.shape}, "
                f"solubility={solubility.shape}"
            ) from exc

        labels = np.asarray(calibration.labels.img)
        if labels.shape != shape:
            raise RuntimeError(
                f"OpenCL labels/frame shape mismatch for {context.run}: "
                f"{labels.shape} != {shape}"
            )

        gas_constraint = None
        aqueous_constraint = None
        expert_adapter = getattr(calibration, "expert_knowledge_adapter", None)
        if expert_adapter is not None:
            reference_frame = context._prepared_colors[0]
            gas_constraint = expert_adapter.mask_for(reference_frame, "saturation_g")
            aqueous_constraint = expert_adapter.mask_for(
                reference_frame,
                "concentration_aq",
            )

        if self.phase_separation == "residual-gas":
            if len(context._gas_scores) != len(context._prepared_colors):
                raise RuntimeError(
                    f"OpenCL residual score count mismatch for {context.run}: "
                    f"{len(context._gas_scores)} != "
                    f"{len(context._prepared_colors)}"
                )

        flags = cl.mem_flags
        resident_bytes = 0

        def readonly(values: Any, dtype: Any) -> tuple[Any, int]:
            array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
            buffer = cl.Buffer(
                self.cl_context,
                flags.READ_ONLY | flags.COPY_HOST_PTR,
                hostbuf=array,
            )
            return buffer, int(array.nbytes)

        lut_y = (
            np.asarray(flash._lut_y, dtype=np.float64)
            if hasattr(flash, "_lut_y")
            else np.array([0.0, 1.0], dtype=np.float64)
        )
        lut_caq = (
            np.asarray(flash._lut_caq, dtype=np.float64)
            if hasattr(flash, "_lut_caq")
            else np.array([0.0, 1.0], dtype=np.float64)
        )
        self.lut_y, size = readonly(lut_y, np.float64)
        resident_bytes += size
        self.lut_caq, size = readonly(lut_caq, np.float64)
        resident_bytes += size
        self.lut_count = int(lut_y.size)

        weighted_g = voxel_volume * density
        weighted_aq = voxel_volume * solubility
        signal_models = calibration.signal_model.model[1]
        self.label_data: dict[int, dict[str, Any]] = {}
        for label in context.signal_labels:
            mask = labels == int(label)
            pixel_count = int(np.count_nonzero(mask))
            if pixel_count == 0:
                continue
            model = signal_models[int(label)]
            supports = np.asarray(model.supports, dtype=np.float64)
            values = np.asarray(model.values, dtype=np.float64)
            if supports.shape != values.shape:
                raise RuntimeError(
                    f"OpenCL signal model shape mismatch for label {label}: "
                    f"{supports.shape} != {values.shape}"
                )

            gas_mask = (
                np.asarray(gas_constraint, dtype=bool)[mask]
                if gas_constraint is not None
                else np.ones(pixel_count, dtype=bool)
            )
            aqueous_mask = (
                np.asarray(aqueous_constraint, dtype=bool)[mask]
                if aqueous_constraint is not None
                else np.ones(pixel_count, dtype=bool)
            )

            data: dict[str, Any] = {
                "model": model,
                "pixel_count": pixel_count,
                "support_count": int(supports.size),
                "signals": [],
                "gas_scores": [],
            }
            for name, host_values, dtype in (
                ("supports", supports, np.float64),
                ("values", values, np.float64),
                ("weight_g", weighted_g[mask], np.float64),
                ("weight_aq", weighted_aq[mask], np.float64),
                ("gas_constraint", gas_mask, np.uint8),
                ("aqueous_constraint", aqueous_mask, np.uint8),
            ):
                data[name], size = readonly(host_values, dtype)
                resident_bytes += size

            for frame in context._prepared_colors:
                buffer, size = readonly(
                    np.asarray(frame.img, dtype=np.float64)[mask],
                    np.float64,
                )
                data["signals"].append(buffer)
                resident_bytes += size

            if self.phase_separation == "residual-gas":
                for score in context._gas_scores:
                    buffer, size = readonly(
                        np.asarray(score, dtype=np.uint8)[mask],
                        np.uint8,
                    )
                    data["gas_scores"].append(buffer)
                    resident_bytes += size
            else:
                data["dummy_score"], size = readonly(
                    np.zeros(pixel_count, dtype=np.uint8),
                    np.uint8,
                )
                resident_bytes += size

            data["output_g"] = cl_array.empty(
                self.queue,
                pixel_count,
                dtype=np.float64,
                allocator=self.allocator,
            )
            data["output_aq"] = cl_array.empty(
                self.queue,
                pixel_count,
                dtype=np.float64,
                allocator=self.allocator,
            )
            resident_bytes += int(data["output_g"].nbytes + data["output_aq"].nbytes)
            self.label_data[int(label)] = data

        self.frame_count = len(context._prepared_colors)
        self.resident_bytes = resident_bytes
        self.queue.finish()
        logger.info(
            "[%s] OpenCL evaluator ready on %s / %s; frames=%d labels=%s "
            "resident_mb=%.0f global_mem_mb=%.0f",
            context.run,
            platform.name,
            device.name,
            self.frame_count,
            sorted(self.label_data),
            resident_bytes / (1024 * 1024),
            int(device.global_mem_size) / (1024 * 1024),
        )

    def evaluate(self, params: dict[str, Any]) -> list[tuple[float, float, float]]:
        cl = self.cl
        cl_array = self.cl_array
        min_aq = float(self.flash.min_value_aq)
        max_aq = float(self.flash.max_value_aq)
        min_g = float(self.flash.min_value_g)
        max_g = float(self.flash.max_value_g)
        residual_onset = float(params.get("gas.residual_onset", 0.20))
        residual_width = max(
            1e-6,
            float(params.get("gas.residual_width", 0.40)),
        )
        residual_mode = self.phase_separation == "residual-gas"

        for data in self.label_data.values():
            values = np.ascontiguousarray(
                np.asarray(data["model"].values, dtype=np.float64)
            )
            if values.size != data["support_count"]:
                raise RuntimeError(
                    "OpenCL model value count changed after context preparation"
                )
            cl.enqueue_copy(self.queue, data["values"], values, is_blocking=False)

        reductions: list[tuple[int, Any, Any]] = []
        for frame_index in range(self.frame_count):
            for data in self.label_data.values():
                pixel_count = data["pixel_count"]
                global_size = ((pixel_count + 255) // 256 * 256,)
                gas_score = (
                    data["gas_scores"][frame_index]
                    if residual_mode
                    else data["dummy_score"]
                )
                self.phase_kernel(
                    self.queue,
                    global_size,
                    (256,),
                    data["signals"][frame_index],
                    data["weight_g"],
                    data["weight_aq"],
                    data["gas_constraint"],
                    data["aqueous_constraint"],
                    gas_score,
                    data["supports"],
                    data["values"],
                    np.int32(data["support_count"]),
                    self.lut_y,
                    self.lut_caq,
                    np.int32(self.lut_count),
                    np.float64(min_aq),
                    np.float64(max_aq),
                    np.float64(min_g),
                    np.float64(max_g),
                    np.float32(residual_onset),
                    np.float32(residual_width),
                    np.float32(self.arrival_scale),
                    np.float32(self.score_scale),
                    np.int32(residual_mode),
                    np.uint64(pixel_count),
                    data["output_g"].data,
                    data["output_aq"].data,
                )
                gas_sum = cl_array.sum(
                    data["output_g"],
                    dtype=np.float64,
                    queue=self.queue,
                )
                aqueous_sum = cl_array.sum(
                    data["output_aq"],
                    dtype=np.float64,
                    queue=self.queue,
                )
                reductions.append((frame_index, gas_sum, aqueous_sum))

        frame_g = np.zeros(self.frame_count, dtype=np.float64)
        frame_aq = np.zeros(self.frame_count, dtype=np.float64)
        for frame_index, gas_sum, aqueous_sum in reductions:
            frame_g[frame_index] += float(gas_sum.get())
            frame_aq[frame_index] += float(aqueous_sum.get())

        return [
            (
                float(frame_g[index] + frame_aq[index]),
                float(frame_g[index]),
                float(frame_aq[index]),
            )
            for index in range(self.frame_count)
        ]

    def close(self) -> None:
        try:
            self.queue.finish()
        except Exception:
            pass
        self.label_data.clear()
        self.lut_y = None
        self.lut_caq = None
        self.phase_kernel = None
        self.program = None
        try:
            self.allocator.stop_holding()
        except Exception:
            pass
        self.allocator = None
        self.queue = None
        self.cl_context = None
        self.context = None
