from __future__ import annotations

from dataclasses import dataclass
import numpy as np


PHASES = ("pack", "unpack", "reach", "wrap", "tighten", "pull", "done")
ACTIONS = ("continue", "tighten", "abort", "retry")


def _clip_pair(u1: float, u2: float) -> np.ndarray:
    return np.array(
        [float(np.clip(u1, 0.0, 1.0)), float(np.clip(u2, 0.0, 1.0))],
        dtype=np.float32,
    )


def _lerp_pair(a: tuple[float, float], b: tuple[float, float], alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return _clip_pair(
        (1.0 - alpha) * a[0] + alpha * b[0],
        (1.0 - alpha) * a[1] + alpha * b[1],
    )


def base_pull_offset(default_mocap_pos, phase: str, phase_time: float, pull_time: float) -> np.ndarray:
    pos = np.asarray(default_mocap_pos, dtype=np.float32).copy()
    if phase != "pull":
        return pos
    alpha = float(np.clip(phase_time / max(pull_time, 1e-6), 0.0, 1.0))
    pos[0] += 0.030 * alpha
    pos[2] += 0.045 * alpha
    return pos


@dataclass
class OpenLoopGraspController:
    # phase durations
    pack_time: float = 0.70
    unpack_time: float = 0.65
    reach_time: float = 1.25
    wrap_time: float = 0.90
    tighten_time: float = 1.10
    pull_time: float = 1.20

    # low-level actuation presets
    neutral_cmd: tuple[float, float] = (0.20, 0.20)

    # pack away from object
    opposite_pack_right: tuple[float, float] = (0.82, 0.18)
    opposite_pack_left: tuple[float, float] = (0.18, 0.82)

    # unpack from opposite bend toward neutral
    # then reach toward object side
    reach_right: tuple[float, float] = (0.18, 0.82)
    reach_left: tuple[float, float] = (0.82, 0.18)

    # stronger curvature around object side
    wrap_right: tuple[float, float] = (0.28, 0.94)
    wrap_left: tuple[float, float] = (0.94, 0.28)

    # final tightening
    tight_right: tuple[float, float] = (0.80, 0.99)
    tight_left: tuple[float, float] = (0.99, 0.80)

    def __post_init__(self):
        self.reset(0.0, 0.0)

    def reset(self, object_x: float, object_z: float) -> None:
        self.object_x = float(object_x)
        self.object_z = float(object_z)
        self.sign = 1.0 if object_x >= 0.0 else -1.0
        self.phase = "pack"
        self.phase_time = 0.0
        self.retry_count = 0

    def _advance(self, next_phase: str) -> None:
        self.phase = next_phase
        self.phase_time = 0.0

    def _phase_targets(self) -> dict[str, tuple[float, float]]:
        if self.sign > 0.0:
            # object is on the right, so pack left first, then go right
            return {
                "pack_opposite": self.opposite_pack_right,
                "reach_side": self.reach_right,
                "wrap_side": self.wrap_right,
                "tight_side": self.tight_right,
            }
        return {
            "pack_opposite": self.opposite_pack_left,
            "reach_side": self.reach_left,
            "wrap_side": self.wrap_left,
            "tight_side": self.tight_left,
        }

    def _maybe_intervene(self, intervention: str | None) -> np.ndarray | None:
        if intervention is None or intervention == "continue":
            return None

        if intervention == "abort":
            self.phase = "done"
            self.phase_time = 0.0
            return _clip_pair(0.02, 0.02)

        if intervention == "retry":
            self.retry_count += 1
            # restart from opposite packing to try the whole maneuver again
            self.phase = "pack"
            self.phase_time = 0.0
            return _clip_pair(*self.neutral_cmd)

        if intervention == "tighten":
            if self.phase in {"wrap", "tighten", "pull"}:
                self.phase = "tighten"
                self.phase_time = 0.0
                return None
            return _clip_pair(0.85, 0.85)

        return None

    def update(self, obs, dt: float, intervention: str | None = None) -> np.ndarray:
        forced = self._maybe_intervene(intervention)
        self.phase_time += float(dt)

        if self.phase == "done":
            return _clip_pair(0.02, 0.02)

        tgt = self._phase_targets()
        pack_opposite = tgt["pack_opposite"]
        reach_side = tgt["reach_side"]
        wrap_side = tgt["wrap_side"]
        tight_side = tgt["tight_side"]

        if self.phase == "pack":
            # First curl away from the object.
            alpha = np.clip(self.phase_time / max(self.pack_time, 1e-6), 0.0, 1.0)
            cmd = _lerp_pair(self.neutral_cmd, pack_opposite, alpha)
            if self.phase_time >= self.pack_time:
                self._advance("unpack")

        elif self.phase == "unpack":
            # Come back from the opposite bend toward neutral/open shape.
            alpha = np.clip(self.phase_time / max(self.unpack_time, 1e-6), 0.0, 1.0)
            cmd = _lerp_pair(pack_opposite, self.neutral_cmd, alpha)
            if self.phase_time >= self.unpack_time:
                self._advance("reach")

        elif self.phase == "reach":
            # Now bend toward the object side.
            alpha = np.clip(self.phase_time / max(self.reach_time, 1e-6), 0.0, 1.0)
            cmd = _lerp_pair(self.neutral_cmd, reach_side, alpha)
            if getattr(obs, "contact", False):
                self._advance("wrap")
            elif self.phase_time >= self.reach_time:
                # still continue to wrap even if contact wasn't seen
                self._advance("wrap")

        elif self.phase == "wrap":
            alpha = np.clip(self.phase_time / max(self.wrap_time, 1e-6), 0.0, 1.0)
            cmd = _lerp_pair(reach_side, wrap_side, alpha)
            if self.phase_time >= self.wrap_time:
                self._advance("tighten")

        elif self.phase == "tighten":
            alpha = np.clip(self.phase_time / max(self.tighten_time, 1e-6), 0.0, 1.0)
            cmd = _lerp_pair(wrap_side, tight_side, alpha)
            if self.phase_time >= self.tighten_time:
                self._advance("pull")

        elif self.phase == "pull":
            cmd = _clip_pair(*tight_side)
            if self.phase_time >= self.pull_time:
                self._advance("done")

        else:
            cmd = _clip_pair(0.10, 0.10)

        return forced if forced is not None else cmd


@dataclass
class InterventionPolicy:
    success_continue: float = 0.65
    quality_continue: float = 0.55
    quality_abort: float = 0.18
    confidence_abort: float = 0.10
    trust_warn: float = 0.35

    # EMA smoothing factor for quality signal.
    # Single-window quality predictions are noisy; a rolling EMA of α=0.15
    # gives ~6-window averaging (≈0.3 s at 20 Hz decisions) which smooths
    # transient dips without lagging too far behind real changes.
    quality_ema_alpha: float = 0.15

    # Per-family threshold overrides.
    # Branched and flat objects are geometrically harder to grasp firmly;
    # using the same thresholds as round objects causes premature tighten/abort
    # that displaces the object. Lower thresholds give those families more time.
    FAMILY_THRESHOLDS: dict = None

    def __post_init__(self):
        if self.FAMILY_THRESHOLDS is None:
            self.FAMILY_THRESHOLDS = {
                "round":     {"success_continue": 0.65, "quality_continue": 0.55, "quality_abort": 0.18},
                "sphere":    {"success_continue": 0.65, "quality_continue": 0.55, "quality_abort": 0.18},
                "box":       {"success_continue": 0.60, "quality_continue": 0.50, "quality_abort": 0.15},
                "capsule":   {"success_continue": 0.60, "quality_continue": 0.50, "quality_abort": 0.15},
                "cylinder":  {"success_continue": 0.60, "quality_continue": 0.50, "quality_abort": 0.15},
                "flat":      {"success_continue": 0.55, "quality_continue": 0.45, "quality_abort": 0.12},
                "branched":  {"success_continue": 0.45, "quality_continue": 0.40, "quality_abort": 0.10},
                "irregular": {"success_continue": 0.50, "quality_continue": 0.42, "quality_abort": 0.12},
            }

    def reset(self) -> None:
        self.last_action = "continue"
        self._quality_ema: float | None = None   # initialised on first call

    def _smooth_quality(self, quality: float) -> float:
        """Return EMA-smoothed quality, seeding with the first observed value."""
        if self._quality_ema is None:
            self._quality_ema = quality
        else:
            self._quality_ema = (
                (1.0 - self.quality_ema_alpha) * self._quality_ema
                + self.quality_ema_alpha * quality
            )
        return self._quality_ema

    def _thresholds_for_family(self, family: str | None) -> dict:
        """Return threshold dict for the predicted object family, falling back to defaults."""
        if family and family in self.FAMILY_THRESHOLDS:
            return self.FAMILY_THRESHOLDS[family]
        return {
            "success_continue": self.success_continue,
            "quality_continue": self.quality_continue,
            "quality_abort":    self.quality_abort,
        }

    def decide(self, model_out: dict, obs, phase: str) -> str:
        success_prob = float(np.asarray(model_out.get("success_prob", [0.5]))[0])
        raw_quality  = float(np.asarray(model_out.get("quality",      [0.5]))[0])
        confidence   = float(np.asarray(model_out.get("confidence",   [0.5]))[0])
        trust        = np.asarray(model_out.get("trust", [1.0]), dtype=np.float32)
        trust_mean   = float(trust.mean()) if trust.size else 1.0

        # Smooth noisy quality predictions with EMA
        quality = self._smooth_quality(raw_quality)

        # Resolve per-family thresholds (uses predicted family if provided)
        family_logits = model_out.get("family_logits", None)
        predicted_family: str | None = None
        if family_logits is not None:
            fa = np.asarray(family_logits)
            if fa.ndim >= 1 and fa.size > 0:
                # Map logit index → family name using the standard object_library order
                _FAMILY_NAMES = ["round", "elongated", "square", "flat", "irregular",
                                  "sphere", "box", "capsule", "cylinder", "branched"]
                idx = int(np.argmax(fa.ravel()))
                predicted_family = _FAMILY_NAMES[idx] if idx < len(_FAMILY_NAMES) else None

        thr = self._thresholds_for_family(predicted_family)
        thr_success_continue = thr["success_continue"]
        thr_quality_continue = thr["quality_continue"]
        thr_quality_abort    = thr["quality_abort"]

        # ── Phase-gated action filtering ───────────────────────────────────────
        # Prevents inappropriate actions at the wrong grasp phase:
        #   • "retry"  only makes sense before contact is established (reach/wrap)
        #   • "abort"  only makes sense when we know the final grasp quality (pull)
        # Without gating, a single noisy window during reach could trigger an
        # abort before the gripper has even touched the object.
        RETRY_PHASES = {"pack", "unpack", "reach", "wrap"}
        ABORT_PHASES = {"pull"}

        if phase == "pull" and success_prob < 0.20:
            if "pull" in ABORT_PHASES:
                self.last_action = "abort"
                return "abort"

        if quality < thr_quality_abort and confidence < 0.30:
            if phase in RETRY_PHASES:
                self.last_action = "retry"
                return "retry"
            # outside retry-allowed phases: tighten instead of retry
            if phase in {"tighten"}:
                self.last_action = "tighten"
                return "tighten"

        if confidence < self.confidence_abort:
            if phase in ABORT_PHASES:
                self.last_action = "abort"
                return "abort"
            # not yet in pull phase — wait rather than act on low-confidence signal
            self.last_action = "continue"
            return "continue"

        if trust_mean < self.trust_warn:
            self.last_action = "tighten"
            return "tighten"

        if getattr(obs, "contact", False) and quality >= thr_quality_continue:
            self.last_action = "tighten"
            return "tighten"

        if success_prob >= thr_success_continue and quality >= thr_quality_continue:
            self.last_action = "continue"
            return "continue"

        if phase in {"wrap", "tighten"}:
            self.last_action = "tighten"
            return "tighten"

        self.last_action = "continue"
        return "continue"