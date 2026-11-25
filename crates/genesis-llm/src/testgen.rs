//! Generators shared by the in-crate property tests: provider and CLI output
//! as it might arrive when something upstream is broken.

use ephyra::prelude::*;
use ephyra::{gens, BoxGen};
use serde_json::{json, Value};

/// A JSON value of the kind a confused server substitutes for a field:
/// extreme counts, wrong types, empty containers.
pub(crate) fn wild_json() -> BoxGen<Value> {
    gens::frequency(vec![
        (4, gens::int(0u64..=5000).map(|n| json!(n)).boxed()),
        (
            3,
            gens::pick_from(vec![
                json!(u64::MAX),
                json!(u64::MAX - 1),
                json!(i64::MIN),
                json!(-1),
                json!(1e308),
                json!(0.5),
                json!(null),
                json!(true),
                json!({}),
                json!([]),
                json!([1, "two"]),
                json!({"text": "nested", "type": "text"}),
            ])
            .boxed(),
        ),
        (
            2,
            gens::string_len(gens::char(), 0..=12)
                .map(Value::String)
                .boxed(),
        ),
        (
            1,
            gens::pick_from(vec![
                json!("refusal"),
                json!("max_tokens"),
                json!("length"),
                json!("MAX_TOKENS"),
                json!("error"),
                json!("assistant"),
                json!("text"),
            ])
            .boxed(),
        ),
    ])
    .boxed()
}

fn node_count(value: &Value) -> usize {
    1 + match value {
        Value::Array(items) => items.iter().map(node_count).sum(),
        Value::Object(map) => map.values().map(node_count).sum(),
        _ => 0,
    }
}

/// Replace the `target`-th node of `value` (pre-order) with `replacement`.
fn replace_node(value: &mut Value, target: &mut usize, replacement: &Value) -> bool {
    if *target == 0 {
        *value = replacement.clone();
        return true;
    }
    *target -= 1;
    match value {
        Value::Array(items) => items
            .iter_mut()
            .any(|item| replace_node(item, target, replacement)),
        Value::Object(map) => map
            .values_mut()
            .any(|item| replace_node(item, target, replacement)),
        _ => false,
    }
}

/// `template` with up to three of its nodes swapped for wild values: output
/// that is nearly what the parser expects, which is where parsers break.
pub(crate) fn near(template: Value) -> BoxGen<Value> {
    gens::vec((gens::int(0usize..=10_000), wild_json()))
        .len(0..=3)
        .map(move |swaps| {
            let mut value = template.clone();
            for (pick, replacement) in swaps {
                let mut target = pick % node_count(&value);
                replace_node(&mut value, &mut target, &replacement);
            }
            value
        })
        .boxed()
}

/// One generated line per template, in order, as newline-delimited JSON.
pub(crate) fn near_lines(templates: Vec<Value>) -> BoxGen<String> {
    let lines: Vec<BoxGen<Value>> = templates.into_iter().map(near).collect();
    (move |source: &mut Source| {
        lines
            .iter()
            .map(|line| source.draw(line).to_string())
            .collect::<Vec<_>>()
            .join("\n")
    })
    .boxed()
}
