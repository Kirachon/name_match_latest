pub fn init_tracing_from_env() {
    // Bridge log:: macros into tracing so existing code keeps working
    let _ = tracing_log::LogTracer::init();
    // Configure subscriber from RUST_LOG or default to info
    let env_filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into());
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .with_level(true)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);
}
