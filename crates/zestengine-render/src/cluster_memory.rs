//! Geometry allocation admission and reclamation policy. No GPU or scene dependencies.

/// Total geometry buffer budget. Required ordinary geometry is reserved first.
pub const DEFAULT_GEOMETRY_MB: u32 = 512;
/// Delay measured in frames/retired slot uses, not time spent waiting on the CPU.
pub(crate) const TRIM_DELAY: u32 = 240;
pub(crate) const POOL_START_BYTES: u64 = 256 << 10;
pub(crate) const STREAM_BLOCK_BYTES: u64 = 1 << 20;

/// Conservative allowance for buffer memory-requirement alignment, per new buffer.
pub(crate) fn allocation_charge(bytes: u64) -> u64 {
    bytes.max(1).saturating_add(65535) & !65535
}

/// A budget rejection is temporary and must not disable the GPU path as a device failure.
#[derive(Debug)]
pub(crate) struct BudgetExceeded;
impl std::fmt::Display for BudgetExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("geometry memory budget: keeping ordinary/coarse geometry and deferring detail")
    }
}
impl std::error::Error for BudgetExceeded {}

/// Peak admission includes the old allocation until its replacement retires.
pub(crate) fn admits(limit: u64, allocated: u64, reserved: u64, additional: u64) -> bool {
    additional == 0 || allocated.saturating_add(reserved).saturating_add(additional) <= limit
}

/// No compaction: only the unused tail may be reclaimed. Keep room for modest growth.
pub(crate) fn trim_target(capacity: u64, high_water: u64, floor: u64) -> Option<u64> {
    let target = high_water.saturating_mul(2).max(floor).max(1).checked_next_power_of_two()?;
    (target <= capacity / 4).then_some(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn growth_accounts_for_reserved_geometry_and_replacement_peak() {
        assert!(admits(100, 40, 20, 40));
        assert!(!admits(100, 40, 20, 41));
        assert!(!admits(100, 80, 0, 30), "old 80-byte allocation still exists during replacement");
        assert!(admits(100, 120, 0, 0), "existing geometry remains usable when mandatory costs exceed budget");
        assert!(!admits(100, u64::MAX, 20, 1));
    }
    #[test]
    fn trimming_preserves_live_offsets_and_uses_hysteresis() {
        assert_eq!(trim_target(4096, 200, 256), Some(512));
        assert_eq!(trim_target(4096, 2500, 256), None, "live allocation near tail pins pool");
        assert_eq!(trim_target(512, 200, 256), None, "do not oscillate around capacity");
        assert_eq!(trim_target(4096, 0, 256), Some(256));
        assert_eq!(trim_target(4096, u64::MAX, 256), None);
    }
}
