//! Creates a wgpu device the way Bevy does (every feature the adapter has), with
//! [`fsr_wgpu::register_device_extensions`] in its creation callback, and reports what changed and whether Vulkan's
//! validation layer complained. Then tries to load FSR, which fails anywhere but Windows.
//!
//! This is the one part of FSR's Windows path that runs on any Vulkan driver, so it is checked here on the
//! drivers at hand. It opens no window.
//!
//! ```sh
//! WGPU_ADAPTER_NAME=AMD cargo run --example device_probe
//! WGPU_ADAPTER_NAME=NVIDIA cargo run --example device_probe -- --any-vendor
//! ```
//!
//! - `--any-vendor`: register on a non-AMD adapter too (FSR's integration only does it on AMD).
//!
//! The validation layer is used when the Vulkan loader finds it (set `VK_LAYER_PATH` if it is not installed
//! system-wide). Exits 1 if it reported an error or the device could not be created.

use std::sync::atomic::{AtomicUsize, Ordering};
use wgpu::hal::api::Vulkan;

static ERRORS: AtomicUsize = AtomicUsize::new(0);

struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            if record.level() == log::Level::Error {
                ERRORS.fetch_add(1, Ordering::Relaxed);
            }
            eprintln!(
                "[{}] {}: {}",
                record.level(),
                record.target(),
                record.args()
            );
        }
    }

    fn flush(&self) {}
}

fn main() {
    log::set_logger(&StderrLogger).unwrap();
    log::set_max_level(log::LevelFilter::Warn);
    let any_vendor = std::env::args().any(|arg| arg == "--any-vendor");

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let wanted = std::env::var("WGPU_ADAPTER_NAME")
        .unwrap_or_default()
        .to_lowercase();
    let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
        .into_iter()
        .find(|adapter| adapter.get_info().name.to_lowercase().contains(&wanted))
        .expect("no Vulkan adapter matches WGPU_ADAPTER_NAME");
    let info = adapter.get_info();
    println!(
        "adapter: {} ({}, {})",
        info.name, info.driver, info.driver_info
    );

    // Bevy's default (`WgpuSettingsPriority::Functionality`): everything the adapter has.
    let mut features = adapter.features();
    if info.device_type == wgpu::DeviceType::DiscreteGpu {
        features.remove(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
    }
    let limits = adapter.limits();
    let descriptor = wgpu::DeviceDescriptor {
        label: Some("device_probe"),
        required_features: features,
        required_limits: limits.clone(),
        // SAFETY: as Bevy's renderer does; this probe runs no shader.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    };

    let mut changes = None;
    // SAFETY: the callback only adds extensions and features the adapter supports (`register_device_extensions`'s
    // contract).
    let device = unsafe {
        let raw_adapter = adapter.as_hal::<Vulkan>().expect("not a Vulkan adapter");
        let amd = fsr_wgpu::is_amd(&raw_adapter);
        println!("AMD adapter: {amd}");
        let open = raw_adapter.open_with_callback(
            features,
            &limits,
            &wgpu::MemoryHints::default(),
            Some(Box::new(|mut args| {
                if amd || any_vendor {
                    changes = Some(fsr_wgpu::register_device_extensions(
                        &mut args,
                        &raw_adapter,
                    ));
                }
            })),
        );
        match open {
            Ok(open) => adapter.create_device_from_hal::<Vulkan>(open, &descriptor),
            Err(error) => {
                println!("FAIL: vkCreateDevice: {error}");
                std::process::exit(1);
            }
        }
    };
    let (device, _queue) = match device {
        Ok(device) => device,
        Err(error) => {
            println!("FAIL: wgpu device: {error}");
            std::process::exit(1);
        }
    };
    match &changes {
        Some(changes) => println!("registered: {changes}"),
        None => println!("registered: nothing (not an AMD adapter; --any-vendor registers anyway)"),
    }
    println!(
        "wgpu features that bear on FSR: SHADER_F16 {}, SUBGROUP {}, EXPERIMENTAL_COOPERATIVE_MATRIX {}",
        device.features().contains(wgpu::Features::SHADER_F16),
        device.features().contains(wgpu::Features::SUBGROUP),
        device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX),
    );
    match fsr_wgpu::FsrSdk::new(&device) {
        Ok(sdk) => match sdk.probe() {
            Ok(version) => println!("FSR provider: {} (id {:#x})", version.name, version.id),
            Err(error) => println!("FSR probe failed: {error}"),
        },
        Err(error) => println!("FSR: {error}"),
    }

    drop(device);
    let errors = ERRORS.load(Ordering::Relaxed);
    if errors > 0 {
        println!("FAIL: {errors} error(s) logged");
        std::process::exit(1);
    }
    println!("PASS: device created, no error logged");
}
