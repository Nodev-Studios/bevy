//! Mesh fitting without a renderer: what [`fit`] returns for each kind of mesh, and what the system puts on
//! entities.

use super::*;
use bevy_mesh::{MeshBuilder, Meshable, VertexFormat};
use bevy_shape::{Cuboid, Sphere};

/// `bevy_solari`'s own test of whether it builds an acceleration structure for a mesh
/// (`is_mesh_raytracing_compatible` in `scene/blas.rs`), which is private there.
fn solari_traces(mesh: &Mesh) -> bool {
    let attributes = mesh
        .attributes()
        .map(|(attribute, _)| (attribute.id, attribute.format))
        .eq(TRACED_ATTRIBUTES.map(|attribute| (attribute.id, attribute.format)));
    mesh.primitive_topology() == PrimitiveTopology::TriangleList
        && attributes
        && mesh.indices().is_some()
}

fn copy_of(mesh: &Mesh, needs: MeshRaytracingFlags) -> Mesh {
    match fit(mesh, needs) {
        Ok(Fitted::Copy(copy)) => *copy,
        other => panic!("expected a fitted copy, got {other:?}"),
    }
}

fn assert_traceable(copy: &Mesh, source: &Mesh, needs: MeshRaytracingFlags) {
    assert!(solari_traces(copy));
    assert_eq!(copy.raytracing, needs);
    assert_eq!(copy.count_vertices(), source.count_vertices());
    assert_eq!(
        copy.attribute(Mesh::ATTRIBUTE_POSITION),
        source.attribute(Mesh::ATTRIBUTE_POSITION)
    );
    for (attribute, values) in copy.attributes() {
        assert_eq!(values.len(), copy.count_vertices(), "{}", attribute.name);
        assert_eq!(VertexFormat::from(values), attribute.format);
    }
}

fn triangle() -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
}

#[test]
fn a_primitive_without_tangents_gets_them() {
    let source = Sphere::new(1.0).mesh().build();
    assert!(!source.contains_attribute(Mesh::ATTRIBUTE_TANGENT));
    let before = source.clone();

    let copy = copy_of(&source, MeshRaytracingFlags::OPAQUE);
    assert_traceable(&copy, &source, MeshRaytracingFlags::OPAQUE);
    assert_eq!(
        copy.attribute(Mesh::ATTRIBUTE_UV_0),
        source.attribute(Mesh::ATTRIBUTE_UV_0)
    );
    assert_eq!(copy.indices(), source.indices());
    let Some(VertexAttributeValues::Float32x4(tangents)) = copy.attribute(Mesh::ATTRIBUTE_TANGENT)
    else {
        panic!("no tangents");
    };
    assert!(tangents
        .iter()
        .any(|tangent| *tangent != PLACEHOLDER_TANGENT));
    // The mesh that is drawn is only read.
    assert_eq!(source, before);
    assert!(matches!(copy.asset_usage, RenderAssetUsages::RENDER_WORLD));
}

#[test]
fn a_mesh_without_uvs_gets_constant_uvs_and_tangents() {
    let mut source = Mesh::from(Cuboid::default());
    source.remove_attribute(Mesh::ATTRIBUTE_UV_0);
    let copy = copy_of(&source, MeshRaytracingFlags::OPAQUE);
    assert_traceable(&copy, &source, MeshRaytracingFlags::OPAQUE);
    assert_eq!(
        copy.attribute(Mesh::ATTRIBUTE_UV_0),
        Some(&VertexAttributeValues::Float32x2(vec![
            [0.0; 2];
            source
                .count_vertices()
        ]))
    );
    assert_eq!(
        copy.attribute(Mesh::ATTRIBUTE_TANGENT),
        Some(&VertexAttributeValues::Float32x4(vec![
            PLACEHOLDER_TANGENT;
            source
                .count_vertices()
        ]))
    );
}

#[test]
fn vertex_colours_and_a_second_uv_set_stay_on_the_drawn_mesh_only() {
    let mut source = Mesh::from(Cuboid::default());
    let vertices = source.count_vertices();
    source.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0, 0.0, 0.0, 1.0]; vertices]);
    source.insert_attribute(Mesh::ATTRIBUTE_UV_1, vec![[0.5, 0.5]; vertices]);
    let copy = copy_of(&source, MeshRaytracingFlags::OPAQUE);
    assert_traceable(&copy, &source, MeshRaytracingFlags::OPAQUE);
    assert!(source.contains_attribute(Mesh::ATTRIBUTE_COLOR));
    assert!(source.contains_attribute(Mesh::ATTRIBUTE_UV_1));
}

#[test]
fn a_mesh_without_indices_gets_them() {
    let mut source = Mesh::from(Cuboid::default());
    source.duplicate_vertices();
    assert!(source.indices().is_none());
    let copy = copy_of(&source, MeshRaytracingFlags::OPAQUE);
    assert_traceable(&copy, &source, MeshRaytracingFlags::OPAQUE);
    let vertices = source.count_vertices() as u32;
    assert_eq!(copy.indices(), Some(&Indices::U32((0..vertices).collect())));
    assert!(source.indices().is_none());
}

#[test]
fn a_mesh_without_normals_gets_them() {
    let source = triangle();
    let copy = copy_of(&source, MeshRaytracingFlags::OPAQUE);
    assert_traceable(&copy, &source, MeshRaytracingFlags::OPAQUE);
    assert_eq!(
        copy.attribute(Mesh::ATTRIBUTE_NORMAL),
        Some(&VertexAttributeValues::Float32x3(vec![[0.0, 0.0, 1.0]; 3]))
    );
}

#[test]
fn sixteen_bit_indices_are_kept() {
    let mut source = Mesh::from(Cuboid::default());
    let indices: Vec<u16> = source.indices().unwrap().iter().map(|i| i as u16).collect();
    source.insert_indices(Indices::U16(indices.clone()));
    let copy = copy_of(&source, MeshRaytracingFlags::OPAQUE);
    assert_traceable(&copy, &source, MeshRaytracingFlags::OPAQUE);
    assert_eq!(copy.indices(), Some(&Indices::U16(indices)));
}

#[test]
fn a_mesh_that_fits_is_traced_as_it_is() {
    let source = Mesh::from(Cuboid::default())
        .with_generated_tangents()
        .unwrap();
    assert!(solari_traces(&source));
    assert_eq!(source.raytracing, MeshRaytracingFlags::OPAQUE);
    assert!(matches!(
        fit(&source, MeshRaytracingFlags::OPAQUE),
        Ok(Fitted::AsIs)
    ));
}

/// An alpha-masked material needs the other kind of acceleration structure, which a mesh does not declare by
/// default: `bevy_solari` then drops the entity with one warning.
#[test]
fn a_mesh_that_fits_but_lacks_the_flag_gets_a_copy_with_it() {
    let source = Mesh::from(Cuboid::default())
        .with_generated_tangents()
        .unwrap();
    let copy = copy_of(&source, MeshRaytracingFlags::NON_OPAQUE);
    assert_traceable(&copy, &source, MeshRaytracingFlags::NON_OPAQUE);
    assert_eq!(
        copy.attribute(Mesh::ATTRIBUTE_TANGENT),
        source.attribute(Mesh::ATTRIBUTE_TANGENT)
    );
    assert_eq!(source.raytracing, MeshRaytracingFlags::OPAQUE);
}

#[test]
fn what_cannot_be_fitted_says_why() {
    let unfit = |mesh: &Mesh| match fit(mesh, MeshRaytracingFlags::OPAQUE) {
        Err(reason) => reason,
        Ok(fitted) => panic!("expected a refusal, got {fitted:?}"),
    };

    let lines = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0; 3], [1.0; 3]]);
    assert_eq!(unfit(&lines), Unfit::NotATriangleList);

    let empty = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    assert_eq!(unfit(&empty), Unfit::NoPositions);

    let mut ragged = triangle();
    ragged.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0; 3]; 4]);
    assert_eq!(unfit(&ragged), Unfit::IncompleteTriangle);

    let mut compressed = Mesh::from(Cuboid::default());
    compressed.compress_normals().unwrap();
    assert_eq!(
        unfit(&compressed),
        Unfit::AttributeFormat(Mesh::ATTRIBUTE_NORMAL.name)
    );

    let mut extracted = Mesh::from(Cuboid::default());
    extracted.asset_usage = RenderAssetUsages::RENDER_WORLD;
    extracted.take_gpu_data().unwrap();
    assert_eq!(unfit(&extracted), Unfit::NotInMainWorld);

    let mut short_normals = Mesh::from(Cuboid::default());
    short_normals.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; 3]);
    assert_eq!(unfit(&short_normals), Unfit::Inconsistent);

    let mut stray_index = Mesh::from(Cuboid::default());
    stray_index.insert_indices(Indices::U32(vec![0, 1, 1000]));
    assert_eq!(unfit(&stray_index), Unfit::Inconsistent);

    for reason in [
        Unfit::Inconsistent,
        Unfit::NotATriangleList,
        Unfit::NotInMainWorld,
        Unfit::NoPositions,
        Unfit::AttributeFormat("Vertex_Normal"),
        Unfit::IncompleteTriangle,
    ] {
        assert!(!reason.to_string().is_empty());
    }
}

#[cfg(feature = "solari")]
mod on_entities {
    use super::*;
    use crate::{
        Fallback, GraphicsRung, ResolvedDenoiser, ResolvedGraphics, ResolvedQuality,
        ResolvedUpscaler,
    };
    use bevy_asset::{AssetEvent, Assets, Handle};
    use bevy_ecs::{message::Messages, prelude::*, system::SystemId};
    use bevy_material::AlphaMode;
    use bevy_mesh::{skinning::SkinnedMesh, Mesh3d};
    use bevy_pbr::{MeshMaterial3d, StandardMaterial};
    use bevy_solari::scene::RaytracingMesh3d;

    fn world(rung: GraphicsRung) -> World {
        let mut world = World::new();
        world.init_resource::<Assets<Mesh>>();
        world.init_resource::<Assets<StandardMaterial>>();
        world.init_resource::<Messages<AssetEvent<Mesh>>>();
        world.init_resource::<systems::FittedMeshes>();
        world.init_resource::<SkippedBlendedMeshes>();
        // Registered once: the system must remember when it last ran to tell what changed since.
        let system = world.register_system(systems::fit_meshes);
        world.insert_resource(FitMeshes(system));
        set_rung(&mut world, rung);
        world
    }

    #[derive(Resource)]
    struct FitMeshes(SystemId);

    fn set_rung(world: &mut World, rung: GraphicsRung) {
        let counts = world
            .get_resource::<ResolvedGraphics>()
            .map(|resolved| (resolved.unfitted_meshes, resolved.skipped_deforming_meshes))
            .unwrap_or_default();
        world.insert_resource(ResolvedGraphics {
            rung,
            upscaler: ResolvedUpscaler::Taa,
            denoiser: if rung.is_traced() {
                ResolvedDenoiser::Nrd
            } else {
                ResolvedDenoiser::None
            },
            quality: ResolvedQuality::NativeAa,
            render_scale: 1.0,
            fallbacks: Vec::<Fallback>::new(),
            unfitted_meshes: counts.0,
            skipped_deforming_meshes: counts.1,
        });
    }

    fn run(world: &mut World) {
        let system = world.resource::<FitMeshes>().0;
        world.run_system(system).unwrap();
    }

    fn mesh(world: &mut World, mesh: Mesh) -> Handle<Mesh> {
        world.resource_mut::<Assets<Mesh>>().add(mesh)
    }

    fn material(world: &mut World, alpha_mode: AlphaMode) -> Handle<StandardMaterial> {
        world
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                alpha_mode,
                ..Default::default()
            })
    }

    fn counts(world: &World) -> (u32, u32) {
        let resolved = world.resource::<ResolvedGraphics>();
        (resolved.unfitted_meshes, resolved.skipped_deforming_meshes)
    }

    fn traced_mesh(world: &World, entity: Entity) -> Option<Handle<Mesh>> {
        world
            .get::<RaytracingMesh3d>(entity)
            .map(|mesh| mesh.0.clone())
    }

    #[test]
    fn standard_meshes_get_a_second_asset_and_the_drawn_one_is_untouched() {
        let mut world = world(GraphicsRung::RtFull);
        let cube = mesh(&mut world, Mesh::from(Cuboid::default()));
        let fitting = mesh(
            &mut world,
            Mesh::from(Cuboid::default())
                .with_generated_tangents()
                .unwrap(),
        );
        let lines = mesh(
            &mut world,
            Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default())
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0; 3], [1.0; 3]]),
        );
        let opaque = material(&mut world, AlphaMode::Opaque);
        let masked = material(&mut world, AlphaMode::Mask(0.5));
        let glass = material(&mut world, AlphaMode::Blend);
        let smoke = material(&mut world, AlphaMode::Add);
        let cube_before = world.resource::<Assets<Mesh>>().get(&cube).unwrap().clone();

        let spawn =
            |world: &mut World, mesh: &Handle<Mesh>, material: &Handle<StandardMaterial>| {
                world
                    .spawn((Mesh3d(mesh.clone()), MeshMaterial3d(material.clone())))
                    .id()
            };
        let a = spawn(&mut world, &cube, &opaque);
        let b = spawn(&mut world, &cube, &opaque);
        let cutout = spawn(&mut world, &cube, &masked);
        let as_is = spawn(&mut world, &fitting, &opaque);
        let wire = spawn(&mut world, &lines, &opaque);
        let pane = spawn(&mut world, &cube, &glass);
        let puff = spawn(&mut world, &cube, &smoke);
        let skinned = world
            .spawn((
                Mesh3d(cube.clone()),
                MeshMaterial3d(opaque.clone()),
                SkinnedMesh::default(),
            ))
            .id();
        let own = world
            .spawn((
                Mesh3d(cube.clone()),
                MeshMaterial3d(opaque.clone()),
                RaytracingMesh3d(cube.clone()),
            ))
            .id();
        let no_material = world.spawn(Mesh3d(cube.clone())).id();

        run(&mut world);
        // Markers and counts are settled on the second run.
        run(&mut world);
        assert_eq!(counts(&world), (1, 1));

        // Two entities with one mesh and one kind of material share one fitted asset.
        let fitted = traced_mesh(&world, a).unwrap();
        assert_ne!(fitted, cube);
        assert_eq!(traced_mesh(&world, b), Some(fitted.clone()));
        let meshes = world.resource::<Assets<Mesh>>();
        assert!(solari_traces(meshes.get(&fitted).unwrap()));
        assert_eq!(
            meshes.get(&fitted).unwrap().raytracing,
            MeshRaytracingFlags::OPAQUE
        );
        // The masked material gets its own copy, which declares the kind it needs.
        let fitted_cutout = traced_mesh(&world, cutout).unwrap();
        assert_ne!(fitted_cutout, fitted);
        assert_eq!(
            meshes.get(&fitted_cutout).unwrap().raytracing,
            MeshRaytracingFlags::NON_OPAQUE
        );
        // A mesh that fits is traced as it is.
        assert_eq!(traced_mesh(&world, as_is), Some(fitting.clone()));
        // The drawn mesh is the same asset with the same content.
        assert_eq!(world.get::<Mesh3d>(a).unwrap().0, cube);
        assert_eq!(meshes.get(&cube).unwrap(), &cube_before);

        assert_eq!(
            world.get::<UnfittedMesh>(wire).unwrap().reason,
            Unfit::NotATriangleList
        );
        assert!(traced_mesh(&world, wire).is_none());
        assert!(world.entity(skinned).contains::<SkippedDeformingMesh>());
        assert!(traced_mesh(&world, skinned).is_none());
        // Glass and particles are drawn, not traced as walls, and counted.
        for entity in [pane, puff] {
            assert!(world.entity(entity).contains::<SkippedBlendedMesh>());
            assert!(traced_mesh(&world, entity).is_none());
        }
        assert_eq!(
            *world.resource::<SkippedBlendedMeshes>(),
            SkippedBlendedMeshes(2)
        );
        // The game's own choice and non-standard entities are not the ladder's.
        assert_eq!(traced_mesh(&world, own), Some(cube.clone()));
        assert!(!world.entity(own).contains::<FittedMesh>());
        assert!(traced_mesh(&world, no_material).is_none());

        // Back on Raster nothing of the ladder's is left, and the game's own stays.
        set_rung(&mut world, GraphicsRung::Raster);
        run(&mut world);
        for entity in [a, b, cutout, as_is, wire, skinned, pane, puff] {
            assert!(traced_mesh(&world, entity).is_none());
            assert!(!world.entity(entity).contains::<SkippedBlendedMesh>());
            assert!(!world.entity(entity).contains::<FittedMesh>());
            assert!(!world.entity(entity).contains::<UnfittedMesh>());
            assert!(!world.entity(entity).contains::<SkippedDeformingMesh>());
        }
        assert_eq!(traced_mesh(&world, own), Some(cube.clone()));
        assert_eq!(counts(&world), (0, 0));
        assert_eq!(
            *world.resource::<SkippedBlendedMeshes>(),
            SkippedBlendedMeshes(0)
        );
        assert!(world.resource::<systems::FittedMeshes>().0.is_empty());

        // And a second visit to a traced rung fits them again.
        set_rung(&mut world, GraphicsRung::RtLite);
        run(&mut world);
        run(&mut world);
        assert!(traced_mesh(&world, a).is_some());
        assert_eq!(counts(&world), (1, 1));
    }

    #[test]
    fn a_changed_mesh_is_fitted_again() {
        let mut world = world(GraphicsRung::RtLite);
        let cube = mesh(&mut world, Mesh::from(Cuboid::default()));
        let sphere = mesh(&mut world, Sphere::new(1.0).mesh().build());
        let opaque = material(&mut world, AlphaMode::Opaque);
        let entity = world
            .spawn((Mesh3d(cube.clone()), MeshMaterial3d(opaque.clone())))
            .id();
        run(&mut world);
        let first = traced_mesh(&world, entity).unwrap();

        // The entity points at another mesh.
        world.get_mut::<Mesh3d>(entity).unwrap().0 = sphere.clone();
        run(&mut world);
        run(&mut world);
        let second = traced_mesh(&world, entity).unwrap();
        assert_ne!(second, first);
        let vertices = |world: &World, handle: &Handle<Mesh>| {
            world
                .resource::<Assets<Mesh>>()
                .get(handle)
                .unwrap()
                .count_vertices()
        };
        assert_eq!(vertices(&world, &second), vertices(&world, &sphere));

        // The mesh asset itself is modified.
        *world
            .resource_mut::<Assets<Mesh>>()
            .get_mut(&sphere)
            .unwrap() = triangle();
        world
            .resource_mut::<Messages<AssetEvent<Mesh>>>()
            .write(AssetEvent::Modified { id: sphere.id() });
        run(&mut world);
        run(&mut world);
        let third = traced_mesh(&world, entity).unwrap();
        assert_ne!(third, second);
        assert_eq!(vertices(&world, &third), 3);
    }

    #[test]
    fn an_asset_that_has_not_loaded_is_tried_again() {
        let mut world = world(GraphicsRung::RtFull);
        let opaque = material(&mut world, AlphaMode::Opaque);
        let pending = world.resource::<Assets<Mesh>>().reserve_handle();
        let entity = world
            .spawn((Mesh3d(pending.clone()), MeshMaterial3d(opaque)))
            .id();
        run(&mut world);
        assert!(traced_mesh(&world, entity).is_none());
        assert_eq!(counts(&world), (0, 0));

        world
            .resource_mut::<Assets<Mesh>>()
            .insert(pending.id(), Mesh::from(Cuboid::default()))
            .unwrap();
        run(&mut world);
        assert!(traced_mesh(&world, entity).is_some());
    }
}
