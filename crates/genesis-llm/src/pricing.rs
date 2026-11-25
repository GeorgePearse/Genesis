//! List prices and model traits.
//!
//! The bundled tables are ShinkaEvolve's (`data/*.csv`, Apache-2.0, credited
//! in their headers). `GENESIS_PRICING_FILE` replaces the model table with a
//! CSV of the same columns. A model the table does not know, or knows without
//! a price, costs nothing here and is reported as usage only: it is never an
//! error to call an unpriced model.

use std::sync::LazyLock;

const LLM_CSV: &str = include_str!("../data/llm_pricing.csv");
const EMBEDDING_CSV: &str = include_str!("../data/embedding_pricing.csv");

/// One row of the model table. Prices are USD per token.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelPrice {
    /// Name as callers write it; OpenRouter rows carry the `openrouter/` prefix.
    pub model_name: String,
    /// Name the provider's API takes.
    pub api_model_name: String,
    pub provider: String,
    pub input_price: Option<f64>,
    pub output_price: Option<f64>,
    /// `(input, output, threshold)`: the prices that apply to the whole
    /// request once its input exceeds `threshold` tokens.
    pub tier2: Option<(f64, f64, u64)>,
    pub is_reasoning: bool,
    /// The model only accepts its default temperature while reasoning.
    pub think_temp_fixed: bool,
    /// Reasoning cannot be switched off.
    pub requires_reasoning: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingPrice {
    pub model_name: String,
    pub api_model_name: String,
    pub provider: String,
    pub input_price: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Catalog {
    models: Vec<ModelPrice>,
    embeddings: Vec<EmbeddingPrice>,
}

fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .skip(1)
        .map(|line| line.split(',').map(str::trim).collect())
}

fn price(cell: Option<&&str>) -> Option<f64> {
    let per_million: f64 = cell?.parse().ok()?;
    (per_million.is_finite() && per_million >= 0.0).then_some(per_million / 1e6)
}

fn flag(cell: Option<&&str>) -> bool {
    matches!(
        cell.map(|c| c.to_ascii_lowercase()).as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("y")
    )
}

fn parse_models(csv: &str) -> Vec<ModelPrice> {
    rows(csv)
        .filter(|cells| cells.len() >= 2 && !cells[0].is_empty())
        .map(|cells| {
            let provider = cells[1].to_string();
            let api_model_name = cells[0].to_string();
            let model_name = if provider == "openrouter" {
                format!("openrouter/{api_model_name}")
            } else {
                api_model_name.clone()
            };
            let tier2 = match (price(cells.get(4)), price(cells.get(5)), cells.get(6)) {
                (Some(input), Some(output), Some(threshold)) => threshold
                    .parse::<f64>()
                    .ok()
                    .map(|t| (input, output, t as u64)),
                _ => None,
            };
            ModelPrice {
                model_name,
                api_model_name,
                provider,
                input_price: price(cells.get(2)),
                output_price: price(cells.get(3)),
                tier2,
                is_reasoning: flag(cells.get(7)),
                think_temp_fixed: flag(cells.get(8)),
                requires_reasoning: flag(cells.get(9)),
            }
        })
        .collect()
}

fn parse_embeddings(csv: &str) -> Vec<EmbeddingPrice> {
    rows(csv)
        .filter_map(|cells| {
            let input_price = price(cells.get(2))?;
            let model_name = cells.first()?.to_string();
            let provider = cells.get(1)?.to_string();
            let api_model_name = if provider == "azure" {
                model_name.trim_start_matches("azure-").to_string()
            } else {
                model_name.clone()
            };
            Some(EmbeddingPrice {
                model_name,
                api_model_name,
                provider,
                input_price,
            })
        })
        .collect()
}

impl Catalog {
    /// Build a catalog from CSV text in the bundled column layout.
    pub fn from_csv(models_csv: &str, embeddings_csv: &str) -> Self {
        Self {
            models: parse_models(models_csv),
            embeddings: parse_embeddings(embeddings_csv),
        }
    }

    pub fn models(&self) -> &[ModelPrice] {
        &self.models
    }

    /// Exact `model_name`, else the single row whose API name matches.
    pub fn lookup(&self, model: &str) -> Option<&ModelPrice> {
        if let Some(row) = self.models.iter().find(|row| row.model_name == model) {
            return Some(row);
        }
        let mut by_api = self.models.iter().filter(|row| row.api_model_name == model);
        match (by_api.next(), by_api.next()) {
            (Some(row), None) => Some(row),
            _ => None,
        }
    }

    /// `(input_cost, output_cost)` in USD, or `None` for an unpriced model.
    /// `output_tokens` counts everything billed as output, reasoning included.
    pub fn cost(&self, model: &str, input_tokens: u64, output_tokens: u64) -> Option<(f64, f64)> {
        let row = self.lookup(model)?;
        let (mut input_price, mut output_price) = (row.input_price?, row.output_price?);
        if let Some((tier_input, tier_output, threshold)) = row.tier2 {
            if input_tokens > threshold {
                (input_price, output_price) = (tier_input, tier_output);
            }
        }
        Some((
            input_price * input_tokens as f64,
            output_price * output_tokens as f64,
        ))
    }

    pub fn embedding_cost(&self, model: &str, tokens: u64) -> Option<f64> {
        self.embeddings
            .iter()
            .find(|row| row.model_name == model || row.api_model_name == model)
            .map(|row| row.input_price * tokens as f64)
    }
}

static CATALOG: LazyLock<Catalog> = LazyLock::new(|| {
    let replacement = std::env::var("GENESIS_PRICING_FILE")
        .ok()
        .filter(|path| !path.trim().is_empty())
        .and_then(|path| match std::fs::read_to_string(&path) {
            Ok(text) => Some(text),
            Err(err) => {
                tracing::warn!(
                    "GENESIS_PRICING_FILE {path} could not be read ({err}); using bundled prices"
                );
                None
            }
        });
    Catalog::from_csv(replacement.as_deref().unwrap_or(LLM_CSV), EMBEDDING_CSV)
});

/// The process-wide catalog.
pub fn catalog() -> &'static Catalog {
    &CATALOG
}

/// The bundled catalog, ignoring `GENESIS_PRICING_FILE`.
pub fn bundled_catalog() -> Catalog {
    Catalog::from_csv(LLM_CSV, EMBEDDING_CSV)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn bundled_prices_parse_per_token() {
        let catalog = bundled_catalog();
        assert!(catalog.models().len() > 80);
        let opus = catalog.lookup("claude-opus-5-5").unwrap();
        assert!(
            close(opus.input_price.unwrap(), 4.0e-6) && close(opus.output_price.unwrap(), 20.0e-6)
        );
        assert!(opus.is_reasoning && opus.requires_reasoning);
        let (input, output) = catalog.cost("gpt-4.1-mini", 1_000_000, 500_000).unwrap();
        assert!(close(input, 0.4) && close(output, 0.8));
    }

    #[test]
    fn tier_two_applies_above_the_threshold_only() {
        let catalog = bundled_catalog();
        let below = catalog.cost("claude-haiku-5-5", 100_000, 10).unwrap();
        let above = catalog.cost("claude-haiku-5-5", 100_001, 10).unwrap();
        assert!(close(below.0, 100_000.0 * 0.1e-6) && close(below.1, 10.0 * 0.5e-6));
        assert!(close(above.0, 100_001.0 * 0.5e-6) && close(above.1, 10.0 * 2.5e-6));
    }

    #[test]
    fn openrouter_rows_resolve_by_either_name_and_unpriced_rows_cost_nothing() {
        let catalog = bundled_catalog();
        assert!(catalog.lookup("openrouter/qwen/qwen3-coder").is_some());
        assert!(catalog.lookup("qwen/qwen3-coder").is_some());
        assert!(catalog
            .lookup("openrouter/x-ai/grok-4")
            .unwrap()
            .input_price
            .is_none());
        assert_eq!(catalog.cost("openrouter/x-ai/grok-4", 10, 10), None);
        assert_eq!(catalog.cost("no-such-model", 10, 10), None);
    }

    #[test]
    fn embeddings_price_by_catalog_or_api_name() {
        let catalog = bundled_catalog();
        assert!(close(
            catalog
                .embedding_cost("text-embedding-3-small", 1_000_000)
                .unwrap(),
            0.02
        ));
        assert!(close(
            catalog
                .embedding_cost("azure-text-embedding-3-large", 1_000_000)
                .unwrap(),
            0.13
        ));
        assert_eq!(catalog.embedding_cost("local-hash", 5), None);
    }

    #[test]
    fn a_replacement_table_uses_the_same_columns() {
        let csv = "# comment\nmodel_name,provider,input_price,output_price,input_price_tier2,output_price_tier2,tier_threshold,is_reasoning,think_temp_fixed,requires_reasoning\nmine,local,1.0,2.0,,,,False,0,0\nfree,local,N/A,N/A,,,,False,0,0\n";
        let catalog = Catalog::from_csv(csv, "");
        assert_eq!(catalog.models().len(), 2);
        assert!(close(
            catalog.cost("mine", 2_000_000, 1_000_000).unwrap().0,
            2.0
        ));
        assert_eq!(catalog.cost("free", 1, 1), None);
    }
}
