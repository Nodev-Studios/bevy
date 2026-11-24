use crate::constants::SpdConstants;

impl SpdConstants {
    /// SPD's constants for downsampling a `width` × `height` rectangle at the texture's origin, in 64 × 64
    /// tiles. SPD makes at most 12 mips.
    pub fn new(width: u32, height: u32) -> SpdConstants {
        let tiles_x = width.div_ceil(64);
        let tiles_y = height.div_ceil(64);
        let mips = (width.max(height) as f32).log2().floor().min(12.0) as u32;

        SpdConstants {
            mips,
            num_work_groups: tiles_x * tiles_y,
            work_group_offset: [0, 0],
            render_size: [width, height],
            _padding: [0; 2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spd_setup_basic() {
        let constants = SpdConstants::new(1920, 1080);
        // (1920 - 1) / 64 + 1 = 30 tiles in x, (1080 - 1) / 64 + 1 = 17 in y.
        assert_eq!(constants.num_work_groups, 30 * 17);
        assert_eq!(constants.work_group_offset, [0, 0]);
        assert_eq!(constants.render_size, [1920, 1080]);
        assert_eq!(constants.mips, 10);
    }

    #[test]
    fn spd_setup_small_resolution() {
        let constants = SpdConstants::new(64, 64);
        assert_eq!(constants.num_work_groups, 1);
        assert_eq!(constants.mips, 6);
    }

    #[test]
    fn spd_setup_large_resolution() {
        let constants = SpdConstants::new(4096, 4096);
        assert_eq!(constants.num_work_groups, 64 * 64);
        assert_eq!(constants.mips, 12);
    }
}
