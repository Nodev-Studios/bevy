//! How an upscaler or a denoiser that fails on the render device tells the app.
//!
//! NRD, FSR and TSR's FSR 3.1 are checked before they are offered, but a driver may still refuse them when
//! they first run. Each keeps a [`BackendFailure`] in both the main world and the render world (a resource of
//! its own: [`NrdFailure`](crate::nrd::NrdFailure), [`FsrFailure`](crate::fsr::FsrFailure),
//! [`TsrFsr3Failure`](crate::tsr::TsrFsr3Failure)). The render world reports the first failure there, the
//! backend stops trying, and the main world reads it, so a settings layer (`bevy_graphics`) can step down to
//! something else.
//!
//! Errors the driver or wgpu raise while a backend creates its shaders and pipelines are caught with wgpu error
//! scopes ([`capture_device_errors`]), so they become such a failure instead of reaching the device's
//! uncaptured-error handler, which panics by default.

use alloc::sync::Arc;
use core::{
    future::Future,
    pin::pin,
    task::{Context, Poll, Waker},
};
use std::sync::Mutex;
use tracing::error;

/// The first failure of one backend on this device, shared between the main and the render world.
#[derive(Clone, Default, Debug)]
pub struct BackendFailure(Arc<Mutex<Option<String>>>);

impl BackendFailure {
    /// Records that the backend failed, and why; logs it as an error the first time. Later failures are
    /// ignored.
    pub fn report(&self, backend: &str, reason: impl core::fmt::Display) {
        let mut failure = self.0.lock().unwrap();
        if failure.is_none() {
            let reason = reason.to_string();
            error!("{backend} failed on this device and stops: {reason}");
            *failure = Some(reason);
        }
    }

    /// Why the backend failed, if it did.
    pub fn reason(&self) -> Option<String> {
        self.0.lock().unwrap().clone()
    }

    /// Whether the backend failed.
    pub fn has_failed(&self) -> bool {
        self.0.lock().unwrap().is_some()
    }
}

/// The environment variable that makes backends fail on purpose, for testing what an app does when a driver
/// refuses one: a comma-separated list of `tsr-fsr3` and `fsr`. NRD has its own,
/// `NRD_WGPU_FORCE_DEVICE_ERROR` (in `nrd_wgpu`).
pub const FORCE_FAILURE_VARIABLE: &str = "BEVY_FORCE_BACKEND_FAILURE";

/// Whether [`FORCE_FAILURE_VARIABLE`] names `backend`.
pub(crate) fn is_forced(backend: &str) -> bool {
    std::env::var(FORCE_FAILURE_VARIABLE)
        .is_ok_and(|list| list.split(',').any(|name| name.trim() == backend))
}

/// Makes wgpu raise a validation error, for [`FORCE_FAILURE_VARIABLE`]: a buffer that is both mappable for
/// reading and a storage buffer.
pub(crate) fn provoke_validation_error(device: &wgpu::Device) {
    let _ = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("forced_backend_failure"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
}

/// Runs `f` inside wgpu error scopes for validation, internal and out-of-memory errors, and returns the first
/// error any of them caught. On native wgpu every error is known when its scope is popped, so nothing waits.
pub fn capture_device_errors<R>(
    device: &wgpu::Device,
    f: impl FnOnce() -> R,
) -> (R, Option<String>) {
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = f();
    // Scopes are a stack: the last pushed is popped first.
    let errors = [validation.pop(), internal.pop(), out_of_memory.pop()];
    let mut context = Context::from_waker(Waker::noop());
    let error = errors.into_iter().find_map(|future| {
        let mut future = pin!(future);
        match future.as_mut().poll(&mut context) {
            Poll::Ready(error) => error.map(|error| error.to_string()),
            Poll::Pending => None,
        }
    });
    (result, error)
}
