//! Layered configuration: typed defaults, then an optional YAML file, then
//! `dotted.key=value` overrides from the command line.
//!
//! Unknown keys fail at load and name the nearest valid key, so a typo never
//! silently runs with a default.

use anyhow::{bail, Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use serde_yaml::{Mapping, Value};
use std::path::Path;

/// Resolve a config of type `T` from its defaults, a YAML file and overrides.
pub fn load<T>(file: Option<&Path>, overrides: &[String]) -> Result<T>
where
    T: Default + Serialize + DeserializeOwned,
{
    let schema = serde_yaml::to_value(T::default())?;
    let mut tree = schema.clone();
    if let Some(path) = file {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let layer: Value = serde_yaml::from_str(&raw)
            .with_context(|| format!("parsing config {}", path.display()))?;
        if !layer.is_null() {
            merge(&mut tree, layer, Some(&schema), "")
                .with_context(|| format!("in config {}", path.display()))?;
        }
    }
    for item in overrides {
        apply_override(&mut tree, &schema, item)?;
    }
    serde_yaml::from_value(tree).context("configuration has a value of the wrong type")
}

/// The schema is the default tree. A mapping that is empty by default holds
/// user data rather than schema (`env`, `llm_kwargs`), so any key may be
/// added to it or below it; `None` means "inside such a mapping".
fn is_open(schema: Option<&Value>) -> bool {
    match schema {
        None => true,
        Some(Value::Mapping(map)) => map.is_empty(),
        Some(_) => false,
    }
}

fn child<'a>(schema: Option<&'a Value>, key: &Value) -> Option<&'a Value> {
    match schema {
        Some(Value::Mapping(map)) if !map.is_empty() => map.get(key),
        _ => None,
    }
}

fn merge(base: &mut Value, layer: Value, schema: Option<&Value>, path: &str) -> Result<()> {
    let (Value::Mapping(_), Value::Mapping(_)) = (&*base, &layer) else {
        *base = layer;
        return Ok(());
    };
    let (Value::Mapping(base_map), Value::Mapping(layer_map)) = (base, layer) else {
        unreachable!("both sides were just checked to be mappings");
    };
    let open = is_open(schema);
    for (key, value) in layer_map {
        let key_text = key.as_str().unwrap_or_default().to_string();
        let child_path = if path.is_empty() {
            key_text
        } else {
            format!("{path}.{key_text}")
        };
        let child_schema = child(schema, &key);
        if !open && child_schema.is_none() {
            match schema {
                Some(Value::Mapping(known)) => bail!(unknown_key(&child_path, known)),
                _ => bail!("unknown config key `{child_path}`"),
            }
        }
        match base_map.get_mut(&key) {
            Some(slot) if slot.is_mapping() && value.is_mapping() => {
                merge(slot, value, child_schema, &child_path)?;
            }
            Some(slot) => *slot = value,
            None => {
                base_map.insert(key, value);
            }
        }
    }
    Ok(())
}

fn unknown_key(path: &str, siblings: &Mapping) -> String {
    let leaf = path.rsplit('.').next().unwrap_or(path);
    let names: Vec<&str> = siblings.keys().filter_map(Value::as_str).collect();
    let nearest = names
        .iter()
        .map(|name| (edit_distance(leaf, name), *name))
        .filter(|(distance, name)| *distance <= (name.len() / 2).max(2))
        .min();
    match nearest {
        Some((_, name)) => {
            let prefix = path.strip_suffix(leaf).unwrap_or("");
            format!("unknown config key `{path}`; did you mean `{prefix}{name}`?")
        }
        None => format!(
            "unknown config key `{path}`; valid keys here: {}",
            names.join(", ")
        ),
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut previous = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { previous } else { previous + 1 };
            previous = row[j + 1];
            row[j + 1] = cost.min(row[j] + 1).min(previous + 1);
        }
    }
    row[b.len()]
}

/// Apply one `a.b.c=value` override. The value is parsed as YAML, so
/// `x=3`, `x=true`, `x=[a,b]`, `x={k: v}` and `x=null` all work; anything
/// that is not valid YAML is taken as a string.
pub fn apply_override(tree: &mut Value, schema: &Value, item: &str) -> Result<()> {
    let Some((key, raw)) = item.split_once('=') else {
        bail!("override `{item}` is not of the form key=value");
    };
    let key = key.trim();
    if key.is_empty() {
        bail!("override `{item}` has an empty key");
    }
    let value: Value = serde_yaml::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
    let segments: Vec<&str> = key.split('.').collect();
    let mut node = tree;
    let mut schema_node = Some(schema);
    let mut walked = String::new();
    for (index, segment) in segments.iter().enumerate() {
        if !walked.is_empty() {
            walked.push('.');
        }
        walked.push_str(segment);
        let map_key = Value::String((*segment).to_string());
        let open = is_open(schema_node);
        let child_schema = child(schema_node, &map_key);
        if !open && child_schema.is_none() {
            match schema_node {
                Some(Value::Mapping(known)) => bail!(unknown_key(&walked, known)),
                _ => bail!("config key `{walked}` cannot be set: its parent is not a mapping"),
            }
        }
        if node.is_null() {
            *node = Value::Mapping(Mapping::new());
        }
        let Value::Mapping(map) = node else {
            bail!("config key `{walked}` cannot be set: its parent is not a mapping");
        };
        let last = index + 1 == segments.len();
        if last {
            map.insert(map_key, value);
            return Ok(());
        }
        node = map
            .entry(map_key)
            .or_insert_with(|| Value::Mapping(Mapping::new()));
        schema_node = child_schema;
    }
    Ok(())
}

/// The fully resolved configuration, as YAML, for `--dry-run` and receipts.
pub fn to_yaml<T: Serialize>(config: &T) -> String {
    serde_yaml::to_string(config).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    #[serde(default)]
    struct Inner {
        max_idea_cycles: u32,
        models: Vec<String>,
        budget: Option<f64>,
    }

    impl Default for Inner {
        fn default() -> Self {
            Self {
                max_idea_cycles: 3,
                models: vec!["a".into()],
                budget: None,
            }
        }
    }

    #[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
    #[serde(default)]
    struct Outer {
        name: String,
        investigation: Inner,
        env: BTreeMap<String, String>,
        kwargs: BTreeMap<String, serde_json::Value>,
    }

    #[test]
    fn defaults_file_and_overrides_layer_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("c.yaml");
        std::fs::write(
            &file,
            "name: from_file\ninvestigation:\n  max_idea_cycles: 7\nenv:\n  A: '1'\n",
        )
        .unwrap();
        let config: Outer = load(
            Some(&file),
            &[
                "investigation.max_idea_cycles=9".into(),
                "investigation.models=[x, y]".into(),
                "investigation.budget=2.5".into(),
                "env.B=two".into(),
                "kwargs.sampling.temperatures=[0.0, 1.0]".into(),
            ],
        )
        .unwrap();
        assert_eq!(config.name, "from_file");
        assert_eq!(config.investigation.max_idea_cycles, 9);
        assert_eq!(config.investigation.models, vec!["x", "y"]);
        assert_eq!(config.investigation.budget, Some(2.5));
        assert_eq!(config.env["A"], "1");
        assert_eq!(config.env["B"], "two");
        assert_eq!(
            config.kwargs["sampling"]["temperatures"][1],
            serde_json::json!(1.0)
        );
    }

    #[test]
    fn unknown_keys_fail_and_suggest_the_nearest_name() {
        let err = load::<Outer>(None, &["investigation.max_idea_cycle=2".into()]).unwrap_err();
        assert!(
            err.to_string()
                .contains("did you mean `investigation.max_idea_cycles`"),
            "{err}"
        );
        let err = load::<Outer>(None, &["nonsense=1".into()]).unwrap_err();
        assert!(err.to_string().contains("valid keys here"), "{err}");
        assert!(load::<Outer>(None, &["novalue".into()]).is_err());
    }

    #[test]
    fn wrong_types_are_reported() {
        assert!(load::<Outer>(None, &["investigation.max_idea_cycles=many".into()]).is_err());
    }
}
