use anyhow::{Result, bail};
use clap::Parser;
use name_matcher::benchmarking::{BenchBackend, BenchConfig, BenchDatasetKind, run_benchmark};

#[derive(Debug, Parser)]
#[command(
    name = "gpu_string_bench",
    about = "Benchmark current GPU/string-distance matching path and shadow engines"
)]
struct Args {
    #[arg(long, default_value = "small")]
    dataset: String,
    #[arg(long, default_value = "cpu")]
    backend: String,
    #[arg(long, default_value_t = 2)]
    warmup_runs: usize,
    #[arg(long, default_value_t = 5)]
    measured_runs: usize,
    #[arg(long)]
    json_only: bool,
    #[arg(long)]
    current_only: bool,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let args = Args::parse();
    let dataset = BenchDatasetKind::parse(&args.dataset)
        .ok_or_else(|| anyhow::anyhow!("unknown dataset: {}", args.dataset))?;
    let backend = parse_backend(&args.backend)?;
    let report = run_benchmark(BenchConfig {
        dataset,
        backend,
        warmup_runs: args.warmup_runs,
        measured_runs: args.measured_runs,
        include_unavailable_engines: !args.current_only,
    })?;
    if !args.json_only {
        print_summary(&report);
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn parse_backend(value: &str) -> Result<BenchBackend> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cpu" => Ok(BenchBackend::Cpu),
        "gpu" => Ok(BenchBackend::Gpu),
        other => bail!("unknown backend: {other}; expected cpu or gpu"),
    }
}

fn print_summary(report: &name_matcher::benchmarking::BenchmarkReport) {
    eprintln!(
        "dataset={} backend={} measured_runs={} recommendation={}",
        report.dataset.id,
        report.config.backend,
        report.config.measured_runs,
        report.summary.recommendation
    );
    for engine in &report.engines {
        let availability = match &engine.availability {
            name_matcher::benchmarking::EngineAvailability::Available => "available".to_string(),
            name_matcher::benchmarking::EngineAvailability::Unavailable { reason } => {
                format!("unavailable ({reason})")
            }
            name_matcher::benchmarking::EngineAvailability::Error { reason } => {
                format!("error ({reason})")
            }
        };
        if let Some(timing) = &engine.timings {
            eprintln!(
                "engine={} status={} p50_us={} p95_us={} matches={}",
                engine.engine_id,
                availability,
                timing.measured_total_us.p50,
                timing.measured_total_us.p95,
                engine
                    .output
                    .as_ref()
                    .map(|output| output.matches_emitted)
                    .unwrap_or_default()
            );
        } else {
            eprintln!("engine={} status={}", engine.engine_id, availability);
        }
    }
}
