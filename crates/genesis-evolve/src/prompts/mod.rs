//! Prompt construction for mutation, repair, novelty, the meta scratchpad
//! and prompt evolution.

pub mod templates;

pub use templates::*;

use genesis_core::Program;
use serde_json::Value;

/// The five full-rewrite instruction variants, one of which is drawn for
/// each full-rewrite proposal.
pub const FULL_SYS_FORMATS: [(&str, &str); 5] = [
    ("default", FULL_SYS_FORMAT_DEFAULT),
    ("different_algorithm", FULL_SYS_FORMAT_DIFFERENT),
    ("context_motivated", FULL_SYS_FORMAT_MOTIVATED),
    ("structural_redesign", FULL_SYS_FORMAT_STRUCTURAL),
    ("parametric_design", FULL_SYS_FORMAT_PARAMETRIC),
];

pub const DEFAULT_TASK_SYS_MSG: &str =
    "You are an expert optimization and algorithm design assistant. \
Improve the program while preserving correctness and immutable regions.";

/// Substitute `{key}` placeholders in one pass, so text inserted for one
/// placeholder is never scanned for another (programs contain braces).
pub fn fill(template: &str, values: &[(&str, &str)]) -> String {
    let mut out =
        String::with_capacity(template.len() + values.iter().map(|(_, v)| v.len()).sum::<usize>());
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let replaced = values.iter().find_map(|(key, value)| {
            after
                .strip_prefix(key)
                .and_then(|tail| tail.strip_prefix('}'))
                .map(|tail| (*value, tail))
        });
        match replaced {
            Some((value, tail)) => {
                out.push_str(value);
                rest = tail;
            }
            None => {
                out.push('{');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

pub fn text_feedback_section(feedback: &str) -> String {
    if feedback.trim().is_empty() {
        return String::new();
    }
    format!(
        "\nHere is additional text feedback about the current program:\n\n{}\n",
        feedback.trim()
    )
}

/// The block listing previously evaluated programs with their metrics, or
/// with their failure when showing the ancestry of an incorrect program.
pub fn eval_history(
    programs: &[Program],
    language: &str,
    include_text_feedback: bool,
    correct: bool,
) -> String {
    let mut out = if correct {
        "Here are the performance metrics of a set of previously implemented programs:\n\n"
            .to_string()
    } else {
        "Here is the error output of a set of previously implemented but incorrect programs:\n\n"
            .to_string()
    };
    for (index, program) in programs.iter().enumerate() {
        if index == 0 {
            out.push_str("# Prior programs\n\n");
        }
        out.push_str(&format!("```{language}\n{}\n```\n\n", program.code));
        if correct {
            out.push_str(&format!("Performance metrics:\n{}\n\n", program.perf_str()));
        } else {
            out.push_str("The program is incorrect and does not pass all validation tests.\n\n");
        }
        if include_text_feedback && !program.text_feedback.trim().is_empty() {
            out.push_str(&format!(
                "Text feedback:\n{}\n\n",
                program.text_feedback.trim()
            ));
        }
    }
    out
}

/// One program with its metrics and correctness, for the meta summariser.
pub fn individual_program_msg(
    program: &Program,
    language: &str,
    include_text_feedback: bool,
) -> String {
    let mut out = format!(
        "# Program to Analyze\n\n```{language}\n{}\n```\n\nPerformance metrics:\n{}\n\n",
        program.code,
        program.perf_str()
    );
    out.push_str(if program.correct {
        "The program is correct and passes all validation tests.\n\n"
    } else {
        "The program is incorrect and does not pass all validation tests.\n\n"
    });
    if include_text_feedback && !program.text_feedback.trim().is_empty() {
        out.push_str(&format!(
            "Text feedback:\n{}\n\n",
            program.text_feedback.trim()
        ));
    }
    out
}

pub fn error_output_section(stdout_log: &str, stderr_log: &str) -> String {
    let mut sections = Vec::new();
    if !stdout_log.trim().is_empty() {
        sections.push(format!(
            "### Standard Output (stdout):\n\n```\n{}\n```",
            stdout_log.trim()
        ));
    }
    if !stderr_log.trim().is_empty() {
        sections.push(format!(
            "### Standard Error (stderr):\n\n```\n{}\n```",
            stderr_log.trim()
        ));
    }
    if sections.is_empty() {
        return "\n### Error Output:\n\nNo error output captured.\n".to_string();
    }
    format!("\n{}\n", sections.join("\n\n"))
}

/// The second program shown for a crossover proposal.
pub fn cross_component(inspiration: &Program, language: &str) -> String {
    format!(
        "# Crossover Inspiration Programs\n```{language}\n{}\n```\n\nPerformance metrics: {}\n\n",
        inspiration.code,
        inspiration.perf_str()
    )
}

/// Top programs as shown to the prompt mutator.
pub fn top_programs_block(
    programs: &[Program],
    language: &str,
    include_text_feedback: bool,
) -> String {
    if programs.is_empty() {
        return "No program examples available.".to_string();
    }
    let parts: Vec<String> = programs
        .iter()
        .enumerate()
        .map(|(index, program)| {
            let mut part = format!(
                "## Program {}\n\n```{language}\n{}\n```\n\n**Score**: {:.4}\n",
                index + 1,
                program.code,
                program.combined_score
            );
            if !program.public_metrics.is_empty() {
                let metrics: Vec<String> = program
                    .public_metrics
                    .iter()
                    .map(|(key, value)| match value {
                        Value::Number(n) if n.is_f64() => {
                            format!("{key}={:.4}", n.as_f64().unwrap_or(0.0))
                        }
                        Value::String(text) => format!("{key}={text}"),
                        other => format!("{key}={other}"),
                    })
                    .collect();
                part.push_str(&format!("**Metrics**: {}\n", metrics.join(", ")));
            }
            if include_text_feedback && !program.text_feedback.trim().is_empty() {
                part.push_str(&format!(
                    "\n**Feedback**:\n{}\n",
                    program.text_feedback.trim()
                ));
            }
            part
        })
        .collect();
    parts.join("\n---\n\n")
}

pub fn global_scratchpad_section(scratchpad: Option<&str>) -> String {
    match scratchpad.map(str::trim).filter(|text| !text.is_empty()) {
        Some(text) => format!(
            "# Global Insights from Meta-Review\nThe following insights have been extracted from analyzing the \
             evolution of programs so far. Use these to guide your prompt improvements:\n\n{text}\n\n"
        ),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn program(code: &str, score: f64, correct: bool) -> Program {
        let mut program = Program::new(code, "python");
        program.combined_score = score;
        program.correct = correct;
        program
    }

    #[test]
    fn fill_substitutes_once_and_leaves_other_braces_alone() {
        let filled = fill(
            "```{language}\n{code_content}\n``` {unknown} {",
            &[
                ("language", "rust"),
                ("code_content", "let x = {language};"),
            ],
        );
        assert_eq!(filled, "```rust\nlet x = {language};\n``` {unknown} {");
        assert_eq!(fill("{a}{a}", &[("a", "{a}")]), "{a}{a}");
    }

    #[test]
    fn templates_carry_their_placeholders() {
        for (template, keys) in [
            (
                DIFF_ITER_MSG,
                vec![
                    "language",
                    "code_content",
                    "performance_metrics",
                    "text_feedback_section",
                ],
            ),
            (
                FIX_ITER_MSG,
                vec![
                    "language",
                    "code_content",
                    "text_feedback_section",
                    "error_output_section",
                ],
            ),
            (
                NOVELTY_USER_MSG,
                vec!["language", "existing_code", "proposed_code"],
            ),
            (
                META_STEP3_USER_MSG,
                vec![
                    "global_insights",
                    "previous_recommendations",
                    "best_program_info",
                    "max_recommendations",
                ],
            ),
            (
                PROMPT_EVO_FULL_USER,
                vec![
                    "current_prompt",
                    "global_scratchpad_section",
                    "top_programs",
                ],
            ),
        ] {
            for key in keys {
                assert!(
                    template.contains(&format!("{{{key}}}")),
                    "missing {{{key}}}"
                );
            }
        }
        assert!(
            DIFF_SYS_FORMAT.contains("<<<<<<< SEARCH")
                && DIFF_SYS_FORMAT.contains("EVOLVE-BLOCK-START")
        );
        assert!(FULL_SYS_FORMATS
            .iter()
            .all(|(_, text)| text.contains("<CODE>") && text.contains("{language}")));
    }

    #[test]
    fn history_blocks_match_the_reference_layout() {
        let mut good = program("print(1)", 1.234, true);
        good.public_metrics.insert("n".into(), json!(3));
        good.text_feedback = " fast \n".into();
        let history = eval_history(&[good.clone()], "python", true, true);
        assert_eq!(
            history,
            "Here are the performance metrics of a set of previously implemented programs:\n\n# Prior programs\n\n\
             ```python\nprint(1)\n```\n\nPerformance metrics:\nCombined score to maximize: 1.23\nn: 3\n\n\
             Text feedback:\nfast\n\n"
        );
        let failures = eval_history(&[program("x", 0.0, false)], "python", false, false);
        assert!(
            failures.starts_with("Here is the error output") && failures.contains("is incorrect")
        );
        assert!(individual_program_msg(&good, "python", false).contains("is correct and passes"));
        assert_eq!(
            error_output_section("", " "),
            "\n### Error Output:\n\nNo error output captured.\n"
        );
        assert!(error_output_section("out", "err")
            .contains("### Standard Error (stderr):\n\n```\nerr\n```"));
        assert_eq!(text_feedback_section("  "), "");
        assert!(top_programs_block(&[good], "python", true)
            .contains("**Score**: 1.2340\n**Metrics**: n=3"));
        assert_eq!(
            top_programs_block(&[], "python", false),
            "No program examples available."
        );
        assert_eq!(global_scratchpad_section(Some("  ")), "");
    }
}
