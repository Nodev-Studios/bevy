//! Exercises the apply layer of the graphics ladder on a real renderer: switching presets at run time, and
//! making meshes ray traceable.
//!
//! It runs a list of stages, each a preset held for `--frames` frames, saves a screenshot at the end of each,
//! prints one line per stage, and exits. The exit code is 0 when everything held, 2 when a stage asked for a
//! preset by name and got another, 3 when the mesh-fitting scene is not as expected, 4 when a picture is
//! black.
//!
//! ```text
//! # Raster, RtLite, RtFull and Raster again in one run:
//! cargo run --example gfx_apply --features bevy_graphics,bevy_solari -- --cycle --out /tmp/cycle
//! # One preset, as a machine without ray tracing would see it:
//! cargo run --example gfx_apply --features bevy_graphics,bevy_solari -- --force-no-rt
//! # The mesh-fitting scene under the reference path tracer:
//! cargo run --example gfx_apply --features bevy_graphics,bevy_solari -- --scene fit --reference
//! ```
//!
//! Pick the adapter with the `WGPU_ADAPTER_NAME` environment variable.

use argh::FromArgs;
use bevy::{
    asset::RenderAssetUsages,
    camera::{visibility::NoFrustumCulling, CameraMainTextureUsages},
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::{Diagnostic, DiagnosticPath, DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    graphics::{
        GraphicsCamera, GraphicsCapabilities, GraphicsOverrides, GraphicsPlugin, GraphicsPreset,
        GraphicsSettings, ResolvedGraphics, UpscaleQuality, UpscalerChoice,
    },
    image::Image,
    mesh::{
        skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
        Indices, VertexAttributeValues,
    },
    prelude::*,
    render::{
        diagnostic::RenderDiagnosticsPlugin,
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages, WgpuFeatures},
        settings::{RenderCreation, WgpuSettings},
        view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured},
        RenderPlugin,
    },
    solari::pathtracer::{Pathtracer, PathtracingPlugin},
    window::{PresentMode, WindowResolution},
};
use std::{num::NonZero, path::PathBuf};

/// Runs presets of the graphics ladder one after another and checks what the apply layer did.
#[derive(FromArgs, Resource, Clone)]
struct Args {
    /// the scene: `lit` (a room with shadowed lights, the default) or `fit` (meshes `bevy_solari` cannot
    /// trace as they are)
    #[argh(option, default = "String::from(\"lit\")")]
    scene: String,
    /// the preset of a single-stage run: `auto` (the default), `raster`, `rt-lite` or `rt-full`
    #[argh(option, default = "String::from(\"auto\")")]
    preset: String,
    /// run four stages instead of one: raster, rt-lite, rt-full, raster
    #[argh(switch)]
    cycle: bool,
    /// the upscaler: `auto` (the default), `off`, `tsr`, `dlss` or `fsr`
    #[argh(option, default = "String::from(\"auto\")")]
    upscaler: String,
    /// the internal resolution: `auto` (the default), `native`, `quality`, `balanced` or `performance`
    #[argh(option, default = "String::from(\"auto\")")]
    quality: String,
    /// effects to force off, separated by commas: `shadows`, `ssao`, `ssr`, `contact-shadows`, `bloom`
    #[argh(option, default = "String::new()")]
    off: String,
    /// draw with `bevy_solari`'s reference path tracer instead of a ladder camera
    #[argh(switch)]
    reference: bool,
    /// start the renderer without ray queries, as on a card that has none
    #[argh(switch)]
    force_no_rt: bool,
    /// frames each stage is held before its screenshot (default 120)
    #[argh(option, default = "120")]
    frames: u32,
    /// the window's size as `WIDTHxHEIGHT` (default 1280x720)
    #[argh(option, default = "String::from(\"1280x720\")")]
    res: String,
    /// the folder the screenshots and `report.txt` go to (default `gfx_apply_out`)
    #[argh(option, default = "PathBuf::from(\"gfx_apply_out\")")]
    out: PathBuf,
}

struct Stage {
    name: &'static str,
    preset: GraphicsPreset,
}

#[derive(Resource)]
struct Run {
    stages: Vec<Stage>,
    current: usize,
    frame: u32,
    waiting_for_screenshot: bool,
    exit_code: u8,
    report: Vec<String>,
}

/// The meshes of the `fit` scene, each drawn in a colour no other has.
#[derive(Resource, Default)]
struct FitScene(Vec<FitMesh>);

struct FitMesh {
    name: &'static str,
    /// The colour as (red, green, blue) on or off.
    colour: [bool; 3],
    /// Whether the mesh is in the traced scene: all but the skinned one.
    traced: bool,
}

fn main() -> AppExit {
    let args: Args = argh::from_env();
    let (width, height) = args
        .res
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
        .expect("--res takes WIDTHxHEIGHT");
    let stages = if args.cycle {
        vec![
            Stage {
                name: "1-raster",
                preset: GraphicsPreset::Raster,
            },
            Stage {
                name: "2-rt-lite",
                preset: GraphicsPreset::RtLite,
            },
            Stage {
                name: "3-rt-full",
                preset: GraphicsPreset::RtFull,
            },
            Stage {
                name: "4-raster",
                preset: GraphicsPreset::Raster,
            },
        ]
    } else if args.reference {
        // The traced scene only exists while a traced rung is resolved.
        vec![Stage {
            name: "reference",
            preset: GraphicsPreset::RtFull,
        }]
    } else {
        let (name, preset) = match args.preset.as_str() {
            "auto" => ("auto", GraphicsPreset::Auto),
            "raster" => ("raster", GraphicsPreset::Raster),
            "rt-lite" => ("rt-lite", GraphicsPreset::RtLite),
            "rt-full" => ("rt-full", GraphicsPreset::RtFull),
            other => panic!("unknown preset {other}"),
        };
        vec![Stage { name, preset }]
    };

    let mut overrides = GraphicsOverrides::default();
    for effect in args.off.split(',').filter(|effect| !effect.is_empty()) {
        *match effect {
            "shadows" => &mut overrides.shadows,
            "ssao" => &mut overrides.ssao,
            "ssr" => &mut overrides.ssr,
            "contact-shadows" => &mut overrides.contact_shadows,
            "bloom" => &mut overrides.bloom,
            other => panic!("unknown effect {other}"),
        } = Some(false);
    }
    if args.scene == "fit" {
        // Bloom would smear one mesh's colour over its neighbours.
        overrides.bloom = Some(false);
    }
    let settings = GraphicsSettings {
        preset: stages[0].preset,
        upscaler: match args.upscaler.as_str() {
            "auto" => UpscalerChoice::Auto,
            "off" => UpscalerChoice::Off,
            "tsr" => UpscalerChoice::Tsr,
            "dlss" => UpscalerChoice::Dlss,
            "fsr" => UpscalerChoice::Fsr,
            other => panic!("unknown upscaler {other}"),
        },
        upscale_quality: match args.quality.as_str() {
            "auto" => UpscaleQuality::Auto,
            "native" => UpscaleQuality::NativeAa,
            "quality" => UpscaleQuality::Quality,
            "balanced" => UpscaleQuality::Balanced,
            "performance" => UpscaleQuality::Performance,
            other => panic!("unknown quality {other}"),
        },
        overrides,
        ..default()
    };

    let mut app = App::new();

    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    app.insert_resource(bevy::anti_alias::dlss::DlssProjectId(
        bevy_asset::uuid::uuid!(
            "0b6bbb4a-7a0c-4b0e-8d0b-6d1f0f6a6f0e" // Don't copy paste this - generate your own UUID!
        ),
    ));

    std::fs::create_dir_all(&args.out).expect("the output folder can be created");
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "gfx_apply".into(),
                    resolution: WindowResolution::new(width, height)
                        .with_scale_factor_override(1.0),
                    present_mode: PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            })
            .set(RenderPlugin {
                render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                    disabled_features: args
                        .force_no_rt
                        .then_some(WgpuFeatures::EXPERIMENTAL_RAY_QUERY),
                    ..default()
                })),
                // A stage's first frames must already be drawn with the stage's pipelines.
                synchronous_pipeline_compilation: true,
                ..default()
            })
            // So the frame a setting changes in is the frame it is drawn in.
            .disable::<PipelinedRenderingPlugin>(),
        FrameTimeDiagnosticsPlugin::default(),
        RenderDiagnosticsPlugin,
        GraphicsPlugin,
    ))
    .insert_resource(settings)
    .insert_resource(ClearColor(Color::BLACK))
    .insert_resource(Run {
        stages,
        current: 0,
        frame: 0,
        waiting_for_screenshot: false,
        exit_code: 0,
        report: Vec::new(),
    })
    .init_resource::<FitScene>()
    .add_systems(Startup, (print_capabilities, spawn_camera))
    .add_systems(Update, drive);

    if args.reference {
        app.add_plugins(PathtracingPlugin);
    }
    match args.scene.as_str() {
        "lit" => app.add_systems(Startup, spawn_lit_scene),
        "fit" => app.add_systems(Startup, spawn_fit_scene),
        other => panic!("unknown scene {other}"),
    };

    app.insert_resource(args).run()
}

fn print_capabilities(capabilities: Res<GraphicsCapabilities>) {
    println!(
        "adapter: {} ({:?}, {:?}), ray query: {}, DLSS SR/RR: {}/{}",
        capabilities.adapter_name,
        capabilities.vendor,
        capabilities.device_type,
        capabilities.ray_query,
        capabilities.dlss_super_resolution,
        capabilities.dlss_ray_reconstruction,
    );
}

fn spawn_camera(mut commands: Commands, args: Res<Args>) {
    let (transform, tonemapping) = if args.scene == "fit" {
        // The colours are read back from the screenshot, so nothing may bend them.
        (
            Transform::from_xyz(0.0, 0.0, 12.0).looking_at(Vec3::ZERO, Vec3::Y),
            Tonemapping::None,
        )
    } else {
        (
            Transform::from_xyz(0.0, 2.2, 6.5).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
            Tonemapping::TonyMcMapface,
        )
    };
    let mut camera = commands.spawn((Camera3d::default(), transform, tonemapping));
    if args.reference {
        camera.insert((
            Pathtracer::default(),
            CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
            Msaa::Off,
        ));
    } else {
        camera.insert(GraphicsCamera);
    }
}

/// A closed room with a glossy floor, a few objects, and three lights that cast shadow maps.
fn spawn_lit_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut wall = |size: Vec3, at: Vec3, color: Color, roughness: f32| {
        (
            Mesh3d(meshes.add(Cuboid::from_size(size))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                perceptual_roughness: roughness,
                ..default()
            })),
            Transform::from_translation(at),
        )
    };
    let white = Color::srgb(0.8, 0.8, 0.8);
    // The floor is glossy, so screen-space reflections and traced reflections both show on it.
    commands.spawn(wall(
        Vec3::new(10.0, 0.2, 10.0),
        Vec3::new(0.0, -0.1, 0.0),
        white,
        0.15,
    ));
    commands.spawn(wall(
        Vec3::new(10.0, 0.2, 10.0),
        Vec3::new(0.0, 4.1, 0.0),
        white,
        0.9,
    ));
    commands.spawn(wall(
        Vec3::new(10.0, 4.4, 0.2),
        Vec3::new(0.0, 2.0, -5.1),
        white,
        0.9,
    ));
    commands.spawn(wall(
        Vec3::new(10.0, 4.4, 0.2),
        Vec3::new(0.0, 2.0, 7.1),
        white,
        0.9,
    ));
    commands.spawn(wall(
        Vec3::new(0.2, 4.4, 12.4),
        Vec3::new(-5.1, 2.0, 1.0),
        Color::srgb(0.8, 0.2, 0.2),
        0.9,
    ));
    commands.spawn(wall(
        Vec3::new(0.2, 4.4, 12.4),
        Vec3::new(5.1, 2.0, 1.0),
        Color::srgb(0.2, 0.8, 0.2),
        0.9,
    ));
    commands.spawn(wall(
        Vec3::new(1.2, 1.2, 1.2),
        Vec3::new(-1.6, 0.6, 0.0),
        Color::srgb(0.9, 0.7, 0.2),
        0.6,
    ));
    commands.spawn(wall(
        Vec3::new(0.8, 2.4, 0.8),
        Vec3::new(1.8, 1.2, -1.5),
        Color::srgb(0.3, 0.4, 0.9),
        0.6,
    ));
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.7))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            metallic: 1.0,
            perceptual_roughness: 0.1,
            ..default()
        })),
        Transform::from_xyz(0.5, 0.7, 1.0),
    ));

    commands.spawn((
        PointLight {
            intensity: 600_000.0,
            range: 20.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-2.5, 3.2, 2.0),
    ));
    commands.spawn((
        SpotLight {
            intensity: 2_000_000.0,
            range: 20.0,
            outer_angle: 0.7,
            inner_angle: 0.5,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(2.5, 3.8, 2.5).looking_at(Vec3::new(0.5, 0.0, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 0.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default(),
    ));
}

/// Seven meshes in a row of colours. Six cannot be traced by `bevy_solari` as they are, each for another
/// reason; the seventh is skinned and must be left out of the traced scene.
fn spawn_fit_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut inverse_bindposes: ResMut<Assets<SkinnedMeshInverseBindposes>>,
    mut ambient: ResMut<GlobalAmbientLight>,
    mut scene: ResMut<FitScene>,
) {
    ambient.brightness = 0.0;

    let cube = || Mesh::from(Cuboid::from_length(1.6));
    let vertices = cube().count_vertices();

    let no_tangents = Sphere::new(0.9).mesh().build();
    let mut no_uvs = cube();
    no_uvs.remove_attribute(Mesh::ATTRIBUTE_UV_0);
    let vertex_colours =
        cube().with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0f32; 4]; vertices]);
    let non_indexed = cube().with_duplicated_vertices();
    let mut u16_indexed = cube();
    let indices = u16_indexed.indices().unwrap().iter();
    let indices = Indices::U16(indices.map(|index| index as u16).collect());
    u16_indexed.insert_indices(indices);
    let alpha_mask = Mesh::from(Rectangle::new(1.8, 1.8));
    let skinned = cube()
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(vec![[0; 4]; vertices]),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_JOINT_WEIGHT,
            vec![[1.0f32, 0.0, 0.0, 0.0]; vertices],
        );

    // A checkerboard of holes: where alpha is 0 the mask cuts the quad away.
    let mask = Image::new(
        Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        (0..16)
            .flat_map(|texel| {
                [
                    255,
                    255,
                    255,
                    if (texel + texel / 4) % 2 == 0 { 255 } else { 0 },
                ]
            })
            .collect(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    let mask = images.add(mask);

    // Each mesh only emits, in a colour whose channels are fully on or off, so a pixel names its mesh.
    let all = [
        ("no-tangents", no_tangents, [true, false, false]),
        ("no-uvs", no_uvs, [false, true, false]),
        ("vertex-colours", vertex_colours, [false, false, true]),
        ("non-indexed", non_indexed, [true, true, false]),
        ("u16-indexed", u16_indexed, [true, false, true]),
        ("alpha-mask", alpha_mask, [false, true, true]),
        ("skinned", skinned, [true, true, true]),
    ];
    for (i, (name, mesh, colour)) in all.into_iter().enumerate() {
        let at = Vec3::new((i % 4) as f32 * 3.0 - 4.5, 1.6 - (i / 4) as f32 * 3.2, 0.0);
        // Just over white once the camera's exposure is applied (about 1/1000 at the default EV100).
        let channel = |on: bool| if on { 1100.0 } else { 0.0 };
        let is_mask = name == "alpha-mask";
        let material = materials.add(StandardMaterial {
            base_color: Color::BLACK,
            base_color_texture: is_mask.then(|| mask.clone()),
            alpha_mode: if is_mask {
                AlphaMode::Mask(0.5)
            } else {
                AlphaMode::Opaque
            },
            emissive: LinearRgba::rgb(channel(colour[0]), channel(colour[1]), channel(colour[2])),
            reflectance: 0.0,
            perceptual_roughness: 1.0,
            ..default()
        });
        let mut entity = commands.spawn((Mesh3d(meshes.add(mesh)), MeshMaterial3d(material)));
        if name == "skinned" {
            // A skinned mesh is placed by its joint, not by its own transform.
            let joint = entity
                .commands()
                .spawn(Transform::from_translation(at))
                .id();
            entity.insert((
                SkinnedMesh {
                    inverse_bindposes: inverse_bindposes.add(vec![Mat4::IDENTITY]),
                    joints: vec![joint],
                },
                NoFrustumCulling,
            ));
        } else {
            entity
                .insert(Transform::from_translation(at).with_rotation(Quat::from_rotation_y(0.4)));
        }
        scene.0.push(FitMesh {
            name,
            colour,
            traced: name != "skinned",
        });
    }
}

/// Holds each stage for its frames, then takes its screenshot.
fn drive(mut run: ResMut<Run>, args: Res<Args>, mut commands: Commands) {
    if run.waiting_for_screenshot {
        return;
    }
    run.frame += 1;
    if run.frame < args.frames {
        return;
    }
    run.waiting_for_screenshot = true;
    let path = args
        .out
        .join(format!("{}.png", run.stages[run.current].name));
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path))
        .observe(finish_stage);
}

/// Reports the stage whose screenshot arrived, then starts the next one or exits.
fn finish_stage(
    screenshot: On<ScreenshotCaptured>,
    mut run: ResMut<Run>,
    args: Res<Args>,
    scene: Res<FitScene>,
    resolved: Res<ResolvedGraphics>,
    diagnostics: Res<DiagnosticsStore>,
    mut settings: ResMut<GraphicsSettings>,
    mut exit: MessageWriter<AppExit>,
) {
    let stage = &run.stages[run.current];
    let mut line = format!(
        "stage {}: asked {:?}, resolved {:?}, denoiser {:?}, upscaler {:?} at {:?}",
        stage.name,
        stage.preset,
        resolved.rung,
        resolved.denoiser,
        resolved.upscaler,
        resolved.quality
    );
    let mut exit_code = 0;
    let asked = match stage.preset {
        GraphicsPreset::Auto => None,
        GraphicsPreset::Raster => Some("Raster"),
        GraphicsPreset::RtLite => Some("RtLite"),
        GraphicsPreset::RtFull => Some("RtFull"),
    };
    if asked.is_some_and(|asked| asked != format!("{:?}", resolved.rung)) {
        exit_code = 2;
    }
    for fallback in &resolved.fallbacks {
        line += &format!("\n  because: {fallback}");
    }

    let milliseconds = |path: &'static str| {
        diagnostics
            .get(&DiagnosticPath::new(path))
            .and_then(Diagnostic::average)
    };
    if let Some(frame) = milliseconds("frame_time") {
        line += &format!("\n  frame: {frame:.2} ms");
    }
    let lighting: Vec<_> = ["presample_light_tiles", "world_cache", "lighting"]
        .into_iter()
        .filter_map(|pass| {
            let path = format!("render/solari_lighting/{pass}/elapsed_gpu");
            let ms = diagnostics
                .get(&DiagnosticPath::new(path))
                .and_then(Diagnostic::average)?;
            Some((pass, ms))
        })
        .collect();
    if resolved.rung.is_traced() && !args.reference && !lighting.is_empty() {
        let total: f64 = lighting.iter().map(|(_, ms)| ms).sum();
        line += &format!("\n  solari lighting on the GPU: {total:.3} ms (");
        for (pass, ms) in &lighting {
            line += &format!(" {pass} {ms:.3}");
        }
        line += " )";
    }

    // The picture, as 8-bit sRGB.
    let image = screenshot
        .image
        .clone()
        .try_into_dynamic()
        .unwrap()
        .into_rgb8();
    let lit_pixels = image
        .pixels()
        .filter(|pixel| pixel.0.iter().any(|c| *c > 8))
        .count();
    let lit_share = lit_pixels as f64 / (image.width() * image.height()) as f64;
    line += &format!("\n  pixels that are not black: {:.1} %", lit_share * 100.0);
    if lit_share < 0.01 {
        exit_code = 4;
    }

    if args.scene == "fit" {
        let traced_scene = resolved.rung.is_traced();
        line += &format!(
            "\n  unfitted meshes: {}, skinned or morphed meshes left out: {}",
            resolved.unfitted_meshes, resolved.skipped_deforming_meshes
        );
        if traced_scene && (resolved.unfitted_meshes != 0 || resolved.skipped_deforming_meshes != 1)
        {
            exit_code = 3;
        }
        for mesh in &scene.0 {
            let pixels = image
                .pixels()
                .filter(|pixel| (0..3).all(|c| (pixel.0[c] > 127) == mesh.colour[c]))
                .count();
            // A ladder camera rasterizes what the eye sees, so every mesh shows there. Only the
            // reference path tracer shows the traced scene itself.
            let expected = !args.reference || mesh.traced;
            line += &format!(
                "\n  {:>14}: {pixels:>6} pixels ({})",
                mesh.name,
                match (expected, pixels > 0) {
                    (true, true) => "visible, as expected",
                    (false, false) => "absent, as expected",
                    (true, false) => "MISSING",
                    (false, true) => "VISIBLE, but should not be traced",
                }
            );
            if expected != (pixels > 0) {
                exit_code = 3;
            }
        }
    }

    println!("{line}");
    run.report.push(line);
    run.exit_code = run.exit_code.max(exit_code);

    run.current += 1;
    run.frame = 0;
    run.waiting_for_screenshot = false;
    if let Some(next) = run.stages.get(run.current) {
        settings.preset = next.preset;
    } else {
        std::fs::write(args.out.join("report.txt"), run.report.join("\n") + "\n").unwrap();
        exit.write(match NonZero::new(run.exit_code) {
            Some(code) => AppExit::Error(code),
            None => AppExit::Success,
        });
    }
}
