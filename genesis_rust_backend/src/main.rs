use anyhow::Result;
use genesis_rust_backend::config::EvolutionConfig;
use genesis_rust_backend::core::runner::EvolutionRunner;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cfg = if let Some(idx) = args.iter().position(|a| a == "--config") {
        if let Some(path) = args.get(idx + 1) {
            EvolutionConfig::from_yaml_file(path)?
        } else {
            EvolutionConfig::default()
        }
    } else {
        EvolutionConfig::default()
    };

    let mut runner = EvolutionRunner::new(cfg);
    runner.run()
}
