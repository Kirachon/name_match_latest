use std::sync::{OnceLock, RwLock};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    #[allow(dead_code)]
    pub total_mb: u64,
    pub used_mb: u64,
    pub avail_mb: u64,
}

static SYS: OnceLock<RwLock<System>> = OnceLock::new();

#[inline]
fn sys_handle() -> &'static RwLock<System> {
    SYS.get_or_init(|| {
        RwLock::new(System::new_with_specifics(
            RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
        ))
    })
}

pub fn memory_stats_mb() -> MemoryStats {
    let lock = sys_handle();
    let mut sys = lock.write().expect("sysinfo lock poisoned");
    sys.refresh_memory();
    // sysinfo returns in bytes for v0.37
    let total_mb = sys.total_memory() / (1024 * 1024);
    let avail_mb = sys.available_memory() / (1024 * 1024);
    let used_mb = total_mb.saturating_sub(avail_mb);
    MemoryStats {
        total_mb,
        used_mb,
        avail_mb,
    }
}
