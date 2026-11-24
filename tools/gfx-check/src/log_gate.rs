//! The log gate: which lines of a run's log make the run fail.

/// Removes ANSI colour sequences, which Bevy's console log carries and the testbed's own `log.txt` does not.
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // "ESC [ parameters letter": drop up to and including the letter.
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Why a log line fails the gate, if it does.
fn offence(line: &str, traced: bool) -> Option<&'static str> {
    let lower = line.to_lowercase();
    // The stock deferred path logs this validation warning on every run; it is not ours to gate.
    if lower.contains("@invariant") {
        return None;
    }
    if lower.contains("panicked") {
        return Some("panic");
    }
    if lower.contains("failed to process shader") {
        return Some("shader error");
    }
    // wgpu reports through its uncaptured-error handler ("wgpu error: Validation Error ...") and, for some
    // failures, through an ERROR line of one of its crates.
    let wgpu_error_line = line
        .split_once("ERROR ")
        .is_some_and(|(_, target)| target.trim_start().starts_with("wgpu"));
    if lower.contains("wgpu error") || lower.contains("validation error") || wgpu_error_line {
        return Some("wgpu error");
    }
    // Solari says "... not loaded" on every card without ray tracing. That is the expected state of a raster
    // run and a failure of a traced one.
    if traced && lower.contains("not loaded") {
        return Some("plugin not loaded on a traced run");
    }
    None
}

/// Returns the offending lines of a log, each with its reason and its line number.
pub fn gate(text: &str, traced: bool) -> Vec<String> {
    text.lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let line = strip_ansi(line);
            offence(&line, traced).map(|reason| {
                let shown: String = line.chars().take(240).collect();
                format!("{reason} (line {}): {}", index + 1, shown.trim())
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::gate;

    const INVARIANT: &str = "\u{1b}[2m2026-11-24T10:54:42.858813Z\u{1b}[0m \u{1b}[33m WARN\u{1b}[0m \u{1b}[2mwgpu_core::validation\u{1b}[0m\u{1b}[2m:\u{1b}[0m Vertex shader with entry point vertex outputs a @builtin(position) without the @invariant attribute and is used in a pipeline with Equal.";
    const NOT_LOADED: &str = "2026-11-24T10:54:42.8Z  WARN bevy_solari::realtime: SolariLightingPlugin not loaded. GPU lacks support for required features: Features { .. }.";

    #[test]
    fn the_invariant_warning_passes() {
        assert!(gate(INVARIANT, true).is_empty());
    }

    #[test]
    fn not_loaded_fails_only_a_traced_run() {
        assert!(gate(NOT_LOADED, false).is_empty());
        assert_eq!(gate(NOT_LOADED, true).len(), 1);
    }

    #[test]
    fn panics_shader_errors_and_wgpu_errors_fail() {
        for line in [
            "thread 'main' panicked at examples/graphics/gfx_testbed/main.rs:1:1:",
            "2026-11-24T10:54:42.8Z ERROR bevy_render::render_resource::pipeline_cache: failed to process shader:",
            "2026-11-24T10:54:42.8Z ERROR bevy_render::error_handler: Caught rendering error: Validation Error",
            "\u{1b}[31mERROR\u{1b}[0m \u{1b}[2mwgpu_hal::vulkan::instance\u{1b}[0m: VALIDATION [VUID-1 (0x1)]",
            "wgpu error: Validation Error",
        ] {
            assert_eq!(gate(line, false).len(), 1, "{line}");
        }
    }

    #[test]
    fn ordinary_lines_pass() {
        let log = "2026-11-24T10:47:01.0Z  INFO bevy_pbr::cluster: GPU clustering is supported on this device.\n\
                   2026-11-24T10:47:01.0Z  WARN bevy_asset::io::web: WebAssetPlugin is potentially insecure!\n\
                   2026-11-24T10:47:01.0Z ERROR gfx_testbed: the resolved preset is not the one asked for";
        assert!(gate(log, true).is_empty());
    }
}
