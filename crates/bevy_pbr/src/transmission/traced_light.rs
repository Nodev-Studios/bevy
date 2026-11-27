//! A hook that lets a ray tracer light the surfaces still shaded forward on a traced view.
//!
//! On a view whose opaque surfaces are lit by a ray tracer (`bevy_solari`), the meshes the rasterizer still
//! draws on top (glass in the transmissive pass, blended surfaces in the transparent pass, forward-opaque
//! materials) are shaded by `pbr_functions.wesl` with the rasterized lights. Those have no shadow maps there
//! (the tracer makes its own shadows) and the ambient and environment light are not occluded, so a glass in the
//! shade would glow as if it stood in the sun.
//!
//! The tracer fixes this through three items:
//!
//! - [`TracedForwardLightLayout`], a render-world resource: the layout of one extra bind group
//!   ([`TRACED_FORWARD_LIGHT_BIND_GROUP_INDEX`]) and the shader definitions its shader module needs;
//! - [`TracedForwardLight`] on a render-world view whose forward materials it lights. The view's mesh pipeline
//!   key then has [`MeshPipelineKey::TRACED_FORWARD_LIGHT`](crate::MeshPipelineKey::TRACED_FORWARD_LIGHT), and
//!   its material pipelines get the bind group and the shader definition `TRACED_FORWARD_LIGHT`, which makes
//!   `pbr_functions.wesl` import `bevy_solari::realtime::forward_light`;
//! - [`TracedForwardLightBindGroup`] on that view, the bind group itself. A view that has the marker and not yet
//!   the bind group skips its forward draws that frame.

use bevy_ecs::{
    component::Component,
    query::{Has, ROQueryItem},
    resource::Resource,
    system::{lifetimeless::Read, SystemParamItem},
};
use bevy_render::{
    render_phase::{PhaseItem, RenderCommand, RenderCommandResult, TrackedRenderPass},
    render_resource::{BindGroup, BindGroupLayoutDescriptor},
};
use bevy_shader::ShaderDefVal;

/// The bind group index of [`TracedForwardLightBindGroup`] in a material pipeline, after the material's own.
pub const TRACED_FORWARD_LIGHT_BIND_GROUP_INDEX: usize = 4;

/// The layout of the bind group a ray tracer gives forward-shaded materials, and the shader definitions its
/// shader module needs. Insert it in the render world before `RenderStartup`.
#[derive(Resource, Clone)]
pub struct TracedForwardLightLayout {
    pub layout: BindGroupLayoutDescriptor,
    pub shader_defs: Vec<ShaderDefVal>,
}

/// On a render-world view: its forward-shaded materials take their light from the ray tracer.
#[derive(Component, Clone, Copy, Default)]
pub struct TracedForwardLight;

/// On a render-world view with [`TracedForwardLight`]: this frame's bind group for its forward materials.
#[derive(Component)]
pub struct TracedForwardLightBindGroup(pub BindGroup);

/// Sets [`TracedForwardLightBindGroup`] at index `I` on views with [`TracedForwardLight`]; does nothing on
/// other views, whose pipelines have no such group.
pub struct SetTracedForwardLightBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetTracedForwardLightBindGroup<I> {
    type Param = ();
    type ViewQuery = (
        Has<TracedForwardLight>,
        Option<Read<TracedForwardLightBindGroup>>,
    );
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        (traced, bind_group): ROQueryItem<'w, '_, Self::ViewQuery>,
        _entity: Option<()>,
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        match (traced, bind_group) {
            (false, _) => RenderCommandResult::Success,
            (true, Some(bind_group)) => {
                pass.set_bind_group(I, &bind_group.0, &[]);
                RenderCommandResult::Success
            }
            // The pipeline expects the group: drawing without it is invalid.
            (true, None) => RenderCommandResult::Skip,
        }
    }
}
