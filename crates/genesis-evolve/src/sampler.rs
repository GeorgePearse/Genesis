//! Turns a sampled parent and its inspirations into the system and user
//! messages of one proposal, choosing the patch type on the way.

use crate::prompts::{self, fill};
use anyhow::{ensure, Result};
use genesis_core::{IdeaBundle, Program};
use rand::Rng;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq)]
pub struct SampledPrompt {
    pub system: String,
    pub user: String,
    /// `diff`, `full`, `cross` or `fix`.
    pub patch_type: String,
    /// Which full-rewrite instruction variant was drawn.
    pub variant: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PromptSampler {
    task_sys_msg: Option<String>,
    language: String,
    patch_types: Vec<String>,
    patch_type_probs: Vec<f64>,
    use_text_feedback: bool,
    inspiration_sort_order: String,
}

/// Guidance added to the system message ahead of the format instructions.
#[derive(Debug, Clone, Default)]
pub struct Guidance<'a> {
    /// One or more recommendations from the meta scratchpad.
    pub meta_recommendations: Option<&'a str>,
    /// The hypothesis this proposal must implement.
    pub idea: Option<&'a IdeaBundle>,
    /// Distilled conclusions from earlier cycles.
    pub conclusions: Option<&'a str>,
    /// Replaces the task system message (an evolved prompt).
    pub system_override: Option<&'a str>,
}

/// The idea as shown to the model that implements it.
pub fn idea_block(idea: &IdeaBundle) -> String {
    let mut out = format!(
        "\n\n# Active Idea\nImplement this hypothesis in your edit.\n\nTitle: {}\nHypothesis: {}\n\nImplementation plan:\n",
        idea.title, idea.hypothesis
    );
    for step in &idea.implementation_plan {
        out.push_str(&format!("- {step}\n"));
    }
    out.push_str("\nThe harness checks your edit against these rubrics after it is evaluated:\n");
    for rubric in &idea.target_rubrics {
        out.push_str(&format!(
            "- [{}] {}: {}",
            rubric.kind, rubric.id, rubric.description
        ));
        if let Some(verification) = &rubric.verification {
            let fragments: Vec<String> = verification
                .patterns
                .iter()
                .map(|pattern| format!("`{pattern}`"))
                .collect();
            out.push_str(&format!(
                " (checked by finding these code fragments in your program, comments excluded: {})",
                fragments.join(", ")
            ));
        }
        out.push('\n');
    }
    if !idea.notes_for_agent.is_empty() {
        out.push_str(&format!("\nNotes: {}\n", idea.notes_for_agent));
    }
    out
}

impl PromptSampler {
    pub fn new(
        task_sys_msg: Option<String>,
        language: &str,
        patch_types: Vec<String>,
        patch_type_probs: Vec<f64>,
        use_text_feedback: bool,
        inspiration_sort_order: &str,
    ) -> Result<Self> {
        ensure!(!patch_types.is_empty(), "patch_types is empty");
        ensure!(
            patch_types.len() == patch_type_probs.len(),
            "patch_types has {} entries but patch_type_probs has {}",
            patch_types.len(),
            patch_type_probs.len()
        );
        for patch_type in &patch_types {
            ensure!(
                matches!(patch_type.as_str(), "diff" | "full" | "cross"),
                "unknown patch type `{patch_type}`; expected diff, full or cross"
            );
        }
        let total: f64 = patch_type_probs.iter().sum();
        ensure!(
            (total - 1.0).abs() <= 1e-6 && patch_type_probs.iter().all(|p| *p >= 0.0),
            "patch_type_probs must be non-negative and sum to 1.0, got {total:.6}"
        );
        ensure!(
            matches!(inspiration_sort_order, "ascending" | "chronological" | "none"),
            "inspiration_sort_order is `{inspiration_sort_order}`; expected ascending, chronological or none"
        );
        Ok(Self {
            task_sys_msg: task_sys_msg.filter(|message| !message.trim().is_empty()),
            language: language.to_string(),
            patch_types,
            patch_type_probs,
            use_text_feedback,
            inspiration_sort_order: inspiration_sort_order.to_string(),
        })
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn task_sys_msg(&self) -> &str {
        self.task_sys_msg
            .as_deref()
            .unwrap_or(prompts::BASE_SYSTEM_MSG)
    }

    fn base_system(&self, guidance: &Guidance<'_>) -> String {
        guidance
            .system_override
            .map(str::to_string)
            .unwrap_or_else(|| self.task_sys_msg().to_string())
    }

    /// Draw a patch type; crossover needs a second program, so it is left
    /// out when there are no inspirations.
    fn sample_patch_type(&self, rng: &mut dyn rand::RngCore, has_inspirations: bool) -> String {
        let mut options: Vec<(&str, f64)> = self
            .patch_types
            .iter()
            .zip(&self.patch_type_probs)
            .filter(|(patch_type, _)| has_inspirations || patch_type.as_str() != "cross")
            .map(|(patch_type, probability)| (patch_type.as_str(), *probability))
            .collect();
        if options.is_empty() {
            options = self
                .patch_types
                .iter()
                .zip(&self.patch_type_probs)
                .map(|(t, p)| (t.as_str(), *p))
                .collect();
        }
        let total: f64 = options.iter().map(|(_, p)| p).sum();
        if total <= 0.0 {
            return options[rng.gen_range(0..options.len())].0.to_string();
        }
        let mut point = rng.gen::<f64>() * total;
        for (patch_type, probability) in &options {
            if point < *probability {
                return (*patch_type).to_string();
            }
            point -= probability;
        }
        options[options.len() - 1].0.to_string()
    }

    /// Archive and top-k inspirations merged, de-duplicated and ordered:
    /// by default worst first, so the best program is nearest the task.
    pub fn inspiration_context(&self, archive: &[Program], top_k: &[Program]) -> Vec<Program> {
        let mut seen = HashSet::new();
        let mut merged: Vec<Program> = archive
            .iter()
            .chain(top_k)
            .filter(|program| seen.insert(program.id.clone()))
            .cloned()
            .collect();
        match self.inspiration_sort_order.as_str() {
            "ascending" => merged.sort_by(|a, b| a.combined_score.total_cmp(&b.combined_score)),
            "chronological" => merged.sort_by(|a, b| a.timestamp.total_cmp(&b.timestamp)),
            _ => {}
        }
        merged
    }

    pub fn sample(
        &self,
        rng: &mut dyn rand::RngCore,
        parent: &Program,
        archive_inspirations: &[Program],
        top_k_inspirations: &[Program],
        guidance: &Guidance<'_>,
    ) -> SampledPrompt {
        let has_inspirations = !archive_inspirations.is_empty() || !top_k_inspirations.is_empty();
        let patch_type = self.sample_patch_type(rng, has_inspirations);
        let mut system = self.base_system(guidance);

        if let Some(recommendations) = guidance
            .meta_recommendations
            .filter(|r| !r.trim().is_empty() && *r != "none")
        {
            if patch_type != "cross" {
                system.push_str(
                    "\n\n# Potential Recommendations\nThe following are potential recommendations for the next program generation:\n",
                );
                system.push_str(&format!("\n{recommendations}"));
            }
        }
        if let Some(conclusions) = guidance.conclusions.filter(|text| !text.trim().is_empty()) {
            system.push_str(&format!(
                "\n\n# Distilled Conclusions\nForward-looking notes from earlier cycles. Bias toward directions found \
                 promising and away from those found unpromising:\n\n{}",
                conclusions.trim()
            ));
        }
        if let Some(idea) = guidance.idea {
            system.push_str(&idea_block(idea));
        }

        let mut variant = None;
        let language = [("language", self.language.as_str())];
        match patch_type.as_str() {
            "diff" => system.push_str(prompts::DIFF_SYS_FORMAT),
            "full" => {
                let (name, format) =
                    prompts::FULL_SYS_FORMATS[rng.gen_range(0..prompts::FULL_SYS_FORMATS.len())];
                variant = Some(name.to_string());
                system.push_str(&fill(format, &language));
            }
            _ => system.push_str(&fill(prompts::CROSS_SYS_FORMAT, &language)),
        }

        let context = self.inspiration_context(archive_inspirations, top_k_inspirations);
        let history = if context.is_empty() {
            String::new()
        } else {
            prompts::eval_history(&context, &self.language, self.use_text_feedback, true)
        };
        let feedback = if self.use_text_feedback {
            format!(
                "\n{}",
                prompts::text_feedback_section(&parent.text_feedback)
            )
        } else {
            String::new()
        };
        let performance = parent.perf_str();
        let values = [
            ("language", self.language.as_str()),
            ("code_content", parent.code.as_str()),
            ("performance_metrics", performance.as_str()),
            ("text_feedback_section", feedback.as_str()),
        ];
        let iteration = match patch_type.as_str() {
            "diff" => fill(prompts::DIFF_ITER_MSG, &values),
            "full" => fill(prompts::FULL_ITER_MSG, &values),
            _ => {
                let all: Vec<&Program> = archive_inspirations
                    .iter()
                    .chain(top_k_inspirations)
                    .collect();
                let partner = all[rng.gen_range(0..all.len())];
                format!(
                    "{}\n\n{}",
                    fill(prompts::CROSS_ITER_MSG, &values),
                    prompts::cross_component(partner, &self.language)
                )
            }
        };
        SampledPrompt {
            system,
            user: format!("{history}\n{iteration}"),
            patch_type,
            variant,
        }
    }

    /// The repair prompt used while no correct program exists: the incorrect
    /// program, its error output, and its ancestry oldest first.
    pub fn sample_fix(
        &self,
        incorrect: &Program,
        ancestors: &[Program],
        guidance: &Guidance<'_>,
    ) -> SampledPrompt {
        let mut system = self.base_system(guidance);
        system.push_str(&fill(
            prompts::FIX_SYS_FORMAT,
            &[("language", self.language.as_str())],
        ));
        let history = if ancestors.is_empty() {
            String::new()
        } else {
            prompts::eval_history(ancestors, &self.language, self.use_text_feedback, false)
        };
        let feedback = if self.use_text_feedback && !incorrect.text_feedback.trim().is_empty() {
            format!(
                "\n{}",
                prompts::text_feedback_section(&incorrect.text_feedback)
            )
        } else {
            String::new()
        };
        let errors = prompts::error_output_section(
            incorrect.meta_str("stdout_log").unwrap_or(""),
            incorrect.meta_str("stderr_log").unwrap_or(""),
        );
        let iteration = fill(
            prompts::FIX_ITER_MSG,
            &[
                ("language", self.language.as_str()),
                ("code_content", incorrect.code.as_str()),
                ("text_feedback_section", feedback.as_str()),
                ("error_output_section", errors.as_str()),
            ],
        );
        let user = if history.is_empty() {
            iteration
        } else {
            format!("{history}\n{iteration}")
        };
        SampledPrompt {
            system,
            user,
            patch_type: "fix".to_string(),
            variant: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_core::idea::Rubric;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use serde_json::json;

    fn program(code: &str, score: f64) -> Program {
        let mut program = Program::new(code, "python");
        program.combined_score = score;
        program.correct = true;
        program
    }

    fn sampler(types: &[&str], probs: &[f64]) -> PromptSampler {
        PromptSampler::new(
            Some("Pack circles.".into()),
            "python",
            types.iter().map(|t| t.to_string()).collect(),
            probs.to_vec(),
            true,
            "ascending",
        )
        .unwrap()
    }

    #[test]
    fn construction_validates_types_and_probabilities() {
        let build = |types: &[&str], probs: &[f64]| {
            PromptSampler::new(
                None,
                "python",
                types.iter().map(|t| t.to_string()).collect(),
                probs.to_vec(),
                false,
                "ascending",
            )
        };
        assert!(build(&["diff", "full"], &[0.5, 0.5]).is_ok());
        assert!(build(&["diff", "full"], &[0.5, 0.4]).is_err());
        assert!(build(&["diff"], &[0.5, 0.5]).is_err());
        assert!(build(&["paper"], &[1.0]).is_err());
        assert!(build(&[], &[]).is_err());
        assert_eq!(
            build(&["diff"], &[1.0]).unwrap().task_sys_msg(),
            prompts::BASE_SYSTEM_MSG
        );
    }

    #[test]
    fn a_diff_prompt_has_guidance_before_the_format_and_the_parent_last() {
        let sampler = sampler(&["diff"], &[1.0]);
        let mut parent = program("# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n", 1.5);
        parent.text_feedback = "slow".into();
        parent.public_metrics.insert("n".into(), json!(26));
        let inspirations = [program("best", 9.0), program("worst", 0.5)];
        let idea = IdeaBundle {
            idea_id: "i1".into(),
            title: "Reheat".into(),
            hypothesis: "Reheating escapes optima.".into(),
            implementation_plan: vec!["add reheating".into()],
            target_rubrics: vec![Rubric {
                id: "uses_reheat".into(),
                description: "reheats".into(),
                kind: "implementation".into(),
                ..Rubric::default()
            }],
            notes_for_agent: "stay correct".into(),
        };
        let guidance = Guidance {
            meta_recommendations: Some("Try restarts."),
            idea: Some(&idea),
            conclusions: Some("Greedy placement plateaus."),
            system_override: None,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let prompt = sampler.sample(
            &mut rng,
            &parent,
            &inspirations[..1],
            &inspirations[1..],
            &guidance,
        );
        assert_eq!(prompt.patch_type, "diff");
        let system = &prompt.system;
        let order = [
            "Pack circles.",
            "# Potential Recommendations",
            "Try restarts.",
            "# Distilled Conclusions",
            "# Active Idea",
            "- [implementation] uses_reheat: reheats",
            "You MUST respond using an edit name",
        ];
        let positions: Vec<usize> = order
            .iter()
            .map(|needle| system.find(needle).unwrap_or_else(|| panic!("{needle}")))
            .collect();
        assert!(
            positions.windows(2).all(|pair| pair[0] < pair[1]),
            "{positions:?}"
        );
        // Inspirations worst first, then the parent with its metrics and feedback.
        let user = &prompt.user;
        assert!(user.find("worst").unwrap() < user.find("best").unwrap());
        assert!(user.find("best").unwrap() < user.find("# Current program").unwrap());
        assert!(user.contains("Combined score to maximize: 1.50\nn: 26"));
        assert!(
            user.contains("Here is additional text feedback about the current program:\n\nslow")
        );
        assert!(user.ends_with(
            "IMPORTANT: Do not rewrite the entire program - focus on targeted improvements."
        ));
    }

    #[test]
    fn crossover_needs_inspirations_and_skips_recommendations() {
        let sampler = sampler(&["cross", "full"], &[0.9, 0.1]);
        let parent = program("p", 1.0);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for _ in 0..30 {
            let alone = sampler.sample(&mut rng, &parent, &[], &[], &Guidance::default());
            assert_eq!(alone.patch_type, "full");
            assert!(alone.variant.is_some());
            assert!(
                !alone.system.contains("{language}"),
                "the format placeholder was left unfilled"
            );
            assert!(alone.user.starts_with("\n# Current program"));
        }
        let partner = [program("partner_code", 2.0)];
        let guidance = Guidance {
            meta_recommendations: Some("Try restarts."),
            ..Guidance::default()
        };
        let crossed = (0..40)
            .map(|_| sampler.sample(&mut rng, &parent, &partner, &[], &guidance))
            .find(|prompt| prompt.patch_type == "cross")
            .expect("crossover is drawn with probability 0.9");
        assert!(crossed
            .user
            .contains("# Crossover Inspiration Programs\n```python\npartner_code\n```"));
        assert!(!crossed.system.contains("Potential Recommendations"));
        assert!(crossed.system.contains("perform crossover"));
    }

    #[test]
    fn the_fix_prompt_shows_errors_and_ancestry() {
        let sampler = sampler(&["diff"], &[1.0]);
        let mut broken = program("raise ValueError", 0.0);
        broken.correct = false;
        broken.text_feedback = "circles overlap".into();
        broken
            .metadata
            .insert("stderr_log".into(), json!("Traceback: boom"));
        let ancestor = program("older", 0.0);
        let prompt = sampler.sample_fix(
            &broken,
            &[ancestor],
            &Guidance {
                system_override: Some("Evolved prompt."),
                ..Guidance::default()
            },
        );
        assert_eq!(prompt.patch_type, "fix");
        assert!(prompt
            .system
            .starts_with("Evolved prompt.\nYou are debugging and fixing"));
        assert!(prompt.system.contains("```python"));
        assert!(prompt.user.starts_with(
            "Here is the error output of a set of previously implemented but incorrect programs"
        ));
        assert!(prompt
            .user
            .contains("### Standard Error (stderr):\n\n```\nTraceback: boom\n```"));
        assert!(prompt.user.contains("circles overlap"));
        let bare = sampler.sample_fix(&broken, &[], &Guidance::default());
        assert!(bare.user.starts_with("# Incorrect Program to Fix"));
    }

    #[test]
    fn inspiration_order_follows_the_setting() {
        let mut a = program("a", 3.0);
        let mut b = program("b", 1.0);
        a.timestamp = 1.0;
        b.timestamp = 2.0;
        let order = |setting: &str| -> Vec<String> {
            let sampler = PromptSampler::new(
                None,
                "python",
                vec!["diff".into()],
                vec![1.0],
                false,
                setting,
            )
            .unwrap();
            sampler
                .inspiration_context(&[a.clone()], &[b.clone(), a.clone()])
                .into_iter()
                .map(|p| p.code)
                .collect()
        };
        assert_eq!(order("ascending"), ["b", "a"]);
        assert_eq!(order("chronological"), ["a", "b"]);
        assert_eq!(order("none"), ["a", "b"]);
    }
}
