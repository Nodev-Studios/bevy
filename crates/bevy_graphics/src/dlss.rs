//! DLSS on the ladder's cameras: Super Resolution and Ray Reconstruction with an explicit mode.
//!
//! # What an app built with the `dlss` feature must do
//!
//! Insert a `DlssProjectId` resource **before** `DefaultPlugins`. `DefaultPlugins` then holds the DLSS start-up
//! plugin, which panics without one ("The `dlss` feature is enabled, but `DlssProjectId` was not added").
//! The id is the game's own, so the ladder cannot supply it. The test `dlss_needs_a_project_id` keeps the
//! requirement honest.
//!
//! # What the ladder takes care of
//!
//! - **The mode is always explicit.** Left on its own `Auto`, DLSS renders at the output's resolution below
//!   about two megapixels, whatever the game wanted; the ladder sets the mode from
//!   [`ResolvedGraphics::quality`], so a quality name means one internal resolution.
//! - **The main texture's usages.** Super Resolution clears its output through the driver, which needs
//!   `COPY_DST` on the camera's main texture besides the `STORAGE_BINDING` both features write through.
//!   `bevy_anti_alias` adds both to every camera with DLSS when it prepares the frame, so neither the ladder
//!   nor the game sets them.
//! - Only one of the two features is on a camera at a time, and both are removed when neither is the
//!   resolved upscaler.

#[cfg(doc)]
use crate::ResolvedGraphics;
#[cfg(feature = "dlss")]
pub(crate) use on::*;

#[cfg(feature = "dlss")]
mod on {
    use crate::{ResolvedGraphics, ResolvedQuality, ResolvedUpscaler};
    use bevy_anti_alias::dlss::{
        Dlss, DlssFeature, DlssPerfQualityMode, DlssRayReconstructionFeature,
        DlssSuperResolutionFeature,
    };
    use bevy_ecs::prelude::*;
    use core::marker::PhantomData;

    /// The DLSS mode that renders at a [`ResolvedQuality`]'s internal resolution.
    pub(crate) fn perf_quality_mode(quality: ResolvedQuality) -> DlssPerfQualityMode {
        match quality {
            ResolvedQuality::NativeAa => DlssPerfQualityMode::Dlaa,
            ResolvedQuality::Quality => DlssPerfQualityMode::Quality,
            ResolvedQuality::Balanced => DlssPerfQualityMode::Balanced,
            ResolvedQuality::Performance => DlssPerfQualityMode::Performance,
        }
    }

    /// Puts on the camera the DLSS feature `resolved` asks for, if any, and removes the other.
    pub(crate) fn apply(camera: &mut EntityCommands, resolved: &ResolvedGraphics) {
        set::<DlssSuperResolutionFeature>(
            camera,
            resolved,
            resolved.upscaler == ResolvedUpscaler::DlssSuperResolution,
        );
        set::<DlssRayReconstructionFeature>(
            camera,
            resolved,
            resolved.upscaler == ResolvedUpscaler::DlssRayReconstruction,
        );
    }

    fn set<F: DlssFeature>(camera: &mut EntityCommands, resolved: &ResolvedGraphics, wanted: bool) {
        if wanted {
            camera.insert(Dlss::<F> {
                perf_quality_mode: perf_quality_mode(resolved.quality),
                // Whatever history the feature kept belongs to other settings.
                reset: true,
                _phantom_data: PhantomData,
            });
        } else {
            camera.remove::<Dlss<F>>();
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use bevy_anti_alias::dlss::DlssInitPlugin;
        use bevy_app::App;

        #[test]
        fn every_quality_has_an_explicit_mode() {
            for quality in [
                ResolvedQuality::NativeAa,
                ResolvedQuality::Quality,
                ResolvedQuality::Balanced,
                ResolvedQuality::Performance,
            ] {
                assert!(perf_quality_mode(quality) != DlssPerfQualityMode::Auto);
            }
        }

        /// The trap the module's documentation warns about, kept as a test so the requirement is noticed
        /// if `bevy_anti_alias` ever stops panicking.
        #[test]
        #[should_panic(expected = "DlssProjectId was not added")]
        fn dlss_needs_a_project_id() {
            App::new().add_plugins(DlssInitPlugin);
        }
    }
}
