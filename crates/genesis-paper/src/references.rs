//! References: a model plans the literature queries, the host retrieves the
//! records, and only retrieved records can be cited.
//!
//! Retrieval asks Semantic Scholar's graph API and falls back to OpenAlex.
//! The bibliography is written from the retrieved fields, escaped, never
//! from text a model or a remote service supplied as BibTeX. The query-plan
//! shape follows the Agentic Scientist's `reference_query_plan.json`
//! (Sakana AI, Apache-2.0).

use crate::text::escape_inline;
use anyhow::{bail, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::sync::OnceLock;
use std::time::Duration;

pub const QUERY_CATEGORIES: &[&str] = &[
    "core_method",
    "baseline_comparator",
    "benchmark_context",
    "architectural_shift",
    "result_driven_theme",
];
const MAX_QUERIES: usize = 10;
const MAX_QUERY_CHARS: usize = 200;
/// Identifies the client to the reference services, and nothing else.
const USER_AGENT: &str = "genesis-paper/0.1 (reference retrieval)";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Reference {
    /// `ref.s2.<paperId>` or `ref.openalex.<id>`.
    pub id: String,
    pub citation_key: String,
    pub title: String,
    pub authors: Vec<String>,
    pub year: Option<u32>,
    pub venue: String,
    pub citation_count: u64,
    /// The planned query that found it.
    pub query: String,
    pub source_url: String,
    pub retrieved_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct PlannedQuery {
    pub query: String,
    pub category: String,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct QueryPlan {
    pub central_topic: String,
    pub queries: Vec<PlannedQuery>,
}

/// Check a model-written plan: one to ten distinct queries of at most 200
/// characters, each in a known category.
pub fn validate_query_plan(value: &Value) -> Result<QueryPlan> {
    let Some(items) = value.get("queries").and_then(Value::as_array) else {
        bail!("the query plan has no `queries` array");
    };
    let mut seen = HashSet::new();
    let mut queries = Vec::new();
    for item in items {
        let query = item
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if query.is_empty() {
            bail!("a planned query is empty");
        }
        if query.chars().count() > MAX_QUERY_CHARS {
            bail!("a planned query is longer than {MAX_QUERY_CHARS} characters");
        }
        let category = item
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if !QUERY_CATEGORIES.contains(&category.as_str()) {
            bail!(
                "query category `{category}` is not one of: {}",
                QUERY_CATEGORIES.join(", ")
            );
        }
        if seen.insert(query.to_lowercase()) {
            queries.push(PlannedQuery {
                query,
                category,
                rationale: item
                    .get("rationale")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string(),
            });
        }
    }
    if queries.is_empty() || queries.len() > MAX_QUERIES {
        bail!(
            "the query plan must hold between 1 and {MAX_QUERIES} queries; it has {}",
            queries.len()
        );
    }
    Ok(QueryPlan {
        central_topic: value
            .get("central_topic")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string(),
        queries,
    })
}

fn ascii_word(text: &str) -> String {
    text.chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_lowercase()
}

/// Give every reference a distinct citation key: first author's surname,
/// year, first long word of the title.
pub fn assign_keys(references: &mut [Reference]) {
    let mut taken: HashSet<String> = HashSet::new();
    for reference in references.iter_mut() {
        let surname = reference
            .authors
            .first()
            .and_then(|name| name.split_whitespace().last())
            .map(ascii_word)
            .unwrap_or_default();
        let surname = if surname.is_empty() {
            "anon".to_string()
        } else {
            surname
        };
        let year = reference
            .year
            .map_or("nd".to_string(), |year| year.to_string());
        let word = reference
            .title
            .split_whitespace()
            .map(ascii_word)
            .find(|word| word.len() >= 4)
            .unwrap_or_default();
        let base = format!("{surname}{year}{word}");
        let mut key = base.clone();
        let mut collisions = 0usize;
        while !taken.insert(key.clone()) {
            collisions += 1;
            key = match collisions {
                1..=25 => format!("{base}{}", (b'a' + collisions as u8) as char),
                _ => format!("{base}{collisions}"),
            };
        }
        reference.citation_key = key;
    }
}

/// The bibliography, written from retrieved fields with every value escaped.
pub fn write_bibtex(references: &[Reference]) -> String {
    let mut out = String::new();
    for reference in references {
        let kind = if reference.venue.trim().is_empty() {
            "misc"
        } else {
            "article"
        };
        out.push_str(&format!("@{kind}{{{},\n", reference.citation_key));
        out.push_str(&format!(
            "  title = {{{{{}}}}},\n",
            escape_inline(&reference.title)
        ));
        let names: Vec<String> = reference
            .authors
            .iter()
            .filter(|name| !name.trim().is_empty())
            .map(|name| format!("{{{}}}", escape_inline(name)))
            .collect();
        if !names.is_empty() {
            out.push_str(&format!("  author = {{{}}},\n", names.join(" and ")));
        }
        if let Some(year) = reference.year {
            out.push_str(&format!("  year = {{{year}}},\n"));
        }
        if kind == "article" {
            out.push_str(&format!(
                "  journal = {{{}}},\n",
                escape_inline(&reference.venue)
            ));
        }
        out.push_str("}\n\n");
    }
    out
}

/// The entry keys of a BibTeX file, in order.
pub fn bib_keys(bibtex: &str) -> Vec<String> {
    static ENTRY: OnceLock<Regex> = OnceLock::new();
    let entry = ENTRY
        .get_or_init(|| Regex::new(r"(?m)^@[A-Za-z]+\{\s*([^,\s{}]+)\s*,").expect("valid regex"));
    entry
        .captures_iter(bibtex)
        .map(|found| found[1].to_string())
        .collect()
}

fn normalised_title(title: &str) -> String {
    ascii_word(title)
}

/// Where and how to look references up.
#[derive(Debug, Clone)]
pub struct Retriever {
    http: reqwest::Client,
    semantic_scholar_url: String,
    openalex_url: String,
    retries: u32,
    backoff: Duration,
}

impl Retriever {
    pub fn new(semantic_scholar_url: &str, openalex_url: &str) -> Self {
        let http = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            http,
            semantic_scholar_url: semantic_scholar_url.trim_end_matches('/').to_string(),
            openalex_url: openalex_url.trim_end_matches('/').to_string(),
            retries: 3,
            backoff: Duration::from_millis(400),
        }
    }

    /// GET a JSON document, retrying on rate limits, server errors and
    /// transport failures.
    async fn get_json(
        &self,
        url: &str,
        query: &[(&str, String)],
        api_key: Option<(&str, String)>,
    ) -> Result<Value> {
        let mut last = String::new();
        for attempt in 0..=self.retries {
            if attempt > 0 {
                tokio::time::sleep(self.backoff * attempt).await;
            }
            let mut request = self.http.get(url).query(query);
            if let Some((header, value)) = &api_key {
                request = request.header(*header, value);
            }
            match request.send().await {
                Ok(response) if response.status().is_success() => {
                    return Ok(response.json::<Value>().await?)
                }
                Ok(response) => {
                    let status = response.status();
                    last = format!("HTTP {status}");
                    if !(status.as_u16() == 429 || status.is_server_error()) {
                        break;
                    }
                }
                Err(error) => last = error.to_string(),
            }
        }
        bail!("{url}: {last}")
    }

    async fn semantic_scholar(&self, query: &str, limit: usize) -> Result<Vec<Reference>> {
        let key = std::env::var("S2_API_KEY")
            .ok()
            .filter(|key| !key.trim().is_empty())
            .map(|key| ("x-api-key", key));
        let document = self
            .get_json(
                &format!("{}/graph/v1/paper/search", self.semantic_scholar_url),
                &[
                    ("query", query.to_string()),
                    ("limit", limit.to_string()),
                    (
                        "fields",
                        "paperId,title,year,venue,citationCount,authors,url".to_string(),
                    ),
                ],
                key,
            )
            .await?;
        let now = chrono::Utc::now().to_rfc3339();
        Ok(document["data"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|paper| {
                let id = paper["paperId"].as_str()?;
                let title = paper["title"].as_str()?.trim();
                (!title.is_empty()).then(|| Reference {
                    id: format!("ref.s2.{id}"),
                    citation_key: String::new(),
                    title: title.to_string(),
                    authors: paper["authors"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|author| author["name"].as_str().map(str::to_string))
                        .collect(),
                    year: paper["year"].as_u64().map(|year| year as u32),
                    venue: paper["venue"].as_str().unwrap_or("").trim().to_string(),
                    citation_count: paper["citationCount"].as_u64().unwrap_or(0),
                    query: query.to_string(),
                    source_url: paper["url"].as_str().unwrap_or("").to_string(),
                    retrieved_at: now.clone(),
                })
            })
            .collect())
    }

    async fn openalex(&self, query: &str, limit: usize) -> Result<Vec<Reference>> {
        let document = self
            .get_json(
                &format!("{}/works", self.openalex_url),
                &[
                    ("search", query.to_string()),
                    ("per-page", limit.to_string()),
                ],
                None,
            )
            .await?;
        let now = chrono::Utc::now().to_rfc3339();
        Ok(document["results"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|work| {
                let url = work["id"].as_str()?;
                let id = url.rsplit('/').next().unwrap_or(url);
                let title = work["display_name"]
                    .as_str()
                    .or_else(|| work["title"].as_str())?
                    .trim();
                (!title.is_empty()).then(|| Reference {
                    id: format!("ref.openalex.{id}"),
                    citation_key: String::new(),
                    title: title.to_string(),
                    authors: work["authorships"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|authorship| {
                            authorship["author"]["display_name"]
                                .as_str()
                                .map(str::to_string)
                        })
                        .collect(),
                    year: work["publication_year"].as_u64().map(|year| year as u32),
                    venue: work["primary_location"]["source"]["display_name"]
                        .as_str()
                        .unwrap_or("")
                        .trim()
                        .to_string(),
                    citation_count: work["cited_by_count"].as_u64().unwrap_or(0),
                    query: query.to_string(),
                    source_url: url.to_string(),
                    retrieved_at: now.clone(),
                })
            })
            .collect())
    }

    /// Run every planned query and keep the most cited distinct records, at
    /// most `max_refs`. A query neither service answers contributes nothing;
    /// the failures are returned so the caller can report them.
    pub async fn retrieve(
        &self,
        plan: &QueryPlan,
        max_refs: usize,
    ) -> (Vec<Reference>, Vec<String>) {
        if max_refs == 0 || plan.queries.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let per_query = max_refs.div_ceil(plan.queries.len()) + 1;
        let mut failures = Vec::new();
        let mut found: Vec<Reference> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for planned in &plan.queries {
            let records = match self.semantic_scholar(&planned.query, per_query).await {
                Ok(records) if !records.is_empty() => records,
                primary => {
                    if let Err(error) = primary {
                        failures.push(format!("Semantic Scholar: {error:#}"));
                    }
                    match self.openalex(&planned.query, per_query).await {
                        Ok(records) => records,
                        Err(error) => {
                            failures.push(format!("OpenAlex: {error:#}"));
                            Vec::new()
                        }
                    }
                }
            };
            let mut records = records;
            records.sort_by(|a, b| {
                b.citation_count
                    .cmp(&a.citation_count)
                    .then(a.title.cmp(&b.title))
            });
            for record in records.into_iter().take(per_query) {
                if seen.insert(normalised_title(&record.title)) {
                    found.push(record);
                }
            }
        }
        found.sort_by(|a, b| {
            b.citation_count
                .cmp(&a.citation_count)
                .then(a.title.cmp(&b.title))
        });
        found.truncate(max_refs);
        assign_keys(&mut found);
        (found, failures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn reference(author: &str, year: Option<u32>, title: &str, venue: &str) -> Reference {
        Reference {
            authors: if author.is_empty() {
                Vec::new()
            } else {
                vec![author.to_string(), "Second Author".to_string()]
            },
            year,
            title: title.to_string(),
            venue: venue.to_string(),
            ..Reference::default()
        }
    }

    #[test]
    fn plans_are_validated_and_deduplicated() {
        let plan = validate_query_plan(&json!({
            "central_topic": " circle packing ",
            "queries": [
                {"query": "LLM  program\nevolution", "category": "core_method", "rationale": "the method"},
                {"query": "llm program evolution", "category": "core_method"},
                {"query": "circle packing in a square", "category": "benchmark_context"}
            ]
        }))
        .unwrap();
        assert_eq!(plan.central_topic, "circle packing");
        assert_eq!(plan.queries.len(), 2);
        assert_eq!(plan.queries[0].query, "LLM program evolution");
        assert!(validate_query_plan(&json!({})).is_err());
        assert!(validate_query_plan(&json!({"queries": []})).is_err());
        assert!(
            validate_query_plan(&json!({"queries": [{"query": "x", "category": "gossip"}]}))
                .is_err()
        );
        assert!(validate_query_plan(
            &json!({"queries": [{"query": "y".repeat(201), "category": "core_method"}]})
        )
        .is_err());
        let many: Vec<Value> = (0..11)
            .map(|i| json!({"query": format!("q{i}"), "category": "core_method"}))
            .collect();
        assert!(validate_query_plan(&json!({"queries": many})).is_err());
    }

    #[test]
    fn keys_are_unique_and_the_bibliography_lists_them_in_order() {
        let mut references = vec![
            reference(
                "Ada Lovelace",
                Some(1843),
                "Notes on the Analytical Engine",
                "Scientific Memoirs",
            ),
            reference(
                "Ada Lovelace",
                Some(1843),
                "Notes on a second engine",
                "Scientific Memoirs",
            ),
            reference("", None, "A & B: 100% {unbalanced", ""),
        ];
        assign_keys(&mut references);
        let keys: Vec<&str> = references.iter().map(|r| r.citation_key.as_str()).collect();
        assert_eq!(
            keys,
            [
                "lovelace1843notes",
                "lovelace1843notesb",
                "anonndunbalanced"
            ]
        );
        let bib = write_bibtex(&references);
        assert_eq!(bib_keys(&bib), keys);
        assert!(bib.contains("title = {{A \\& B: 100\\% \\textbraceleft{}unbalanced}},"));
        assert!(bib.contains("@misc{anonndunbalanced,\n  title"));
        assert!(!bib.contains("author = {{}"));
        assert!(bib.contains("author = {{Ada Lovelace} and {Second Author}},"));
        assert!(bib_keys("no entries").is_empty());
    }
}
