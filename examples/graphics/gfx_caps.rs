//! Prints what the graphics ladder knows about this machine, and what its `Auto` settings resolve to here.
//!
//! It opens no window. Pick the adapter with the `WGPU_ADAPTER_NAME` environment variable.
//!
//! ```text
//! cargo run --example gfx_caps --features bevy_graphics,bevy_solari
//! ```

use bevy::{
    app::ScheduleRunnerPlugin,
    graphics::{resolve, GraphicsCapabilities, GraphicsPlugin, GraphicsSettings},
    prelude::*,
    render::{pipelined_rendering::PipelinedRenderingPlugin, RenderPlugin},
    window::ExitCondition,
    winit::WinitPlugin,
};

fn main() {
    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(bevy::anti_alias::dlss::DlssProjectId(
        bevy_asset::uuid::uuid!(
            "e10d2557-5685-4162-bb92-beeae930f009" // Don't copy paste this - generate your own UUID!
        ),
    ));

    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            // The app runs one update and exits. Nothing may still be compiling on another thread then: some
            // drivers crash when the device goes away under a compilation.
            .set(RenderPlugin {
                synchronous_pipeline_compilation: true,
                ..default()
            })
            .disable::<PipelinedRenderingPlugin>()
            // The schedule runner below replaces winit's, so no window is ever created.
            .disable::<WinitPlugin>(),
        ScheduleRunnerPlugin::run_once(),
        GraphicsPlugin,
    ))
    .add_systems(Startup, print_capabilities)
    .run();
}

fn print_capabilities(capabilities: Res<GraphicsCapabilities>) {
    let yes_no = |flag| if flag { "yes" } else { "no" };
    let video_memory = match capabilities.video_memory_bytes {
        Some(bytes) => format!(
            "{bytes} bytes ({:.2} GiB)",
            bytes as f64 / (1u64 << 30) as f64
        ),
        None => "unknown".into(),
    };

    println!("adapter:                  {}", capabilities.adapter_name);
    println!("vendor:                   {:?}", capabilities.vendor);
    println!("device type:              {:?}", capabilities.device_type);
    println!("driver:                   {}", capabilities.driver);
    println!("driver info:              {}", capabilities.driver_info);
    println!("vulkan:                   {}", yes_no(capabilities.vulkan));
    println!("video memory:             {video_memory}");
    println!("ray tracing compiled in:  {}", yes_no(capabilities.solari));
    println!(
        "ray query:                {}",
        yes_no(capabilities.ray_query)
    );
    println!(
        "DLSS Super Resolution:    {}",
        yes_no(capabilities.dlss_super_resolution)
    );
    println!(
        "DLSS Ray Reconstruction:  {}",
        yes_no(capabilities.dlss_ray_reconstruction)
    );
    println!("TSR:                      {}", yes_no(capabilities.tsr));
    println!("FSR:                      {:?}", capabilities.fsr);
    println!("NRD:                      {}", yes_no(capabilities.nrd));

    println!();
    println!("Auto resolves to:");
    for output in [
        UVec2::new(1920, 1080),
        UVec2::new(2560, 1440),
        UVec2::new(3840, 2160),
    ] {
        let resolved = resolve(&GraphicsSettings::default(), &capabilities, output);
        println!(
            "  {}x{}: {:?}, denoiser {:?}, upscaler {:?} at {:?} ({:.0} %)",
            output.x,
            output.y,
            resolved.rung,
            resolved.denoiser,
            resolved.upscaler,
            resolved.quality,
            resolved.render_scale * 100.0
        );
        for fallback in &resolved.fallbacks {
            println!("    because: {fallback}");
        }
    }
}
