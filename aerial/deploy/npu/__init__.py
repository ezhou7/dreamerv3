"""Framework-free policy inference for the Orange Pi 5 (RK3588) NPU.

See aerial/deploy/NPU_PLAN.md. Modules here must not import JAX at
module level -- they run on the companion computer, which only has
numpy (and later rknn-toolkit-lite2).
"""
