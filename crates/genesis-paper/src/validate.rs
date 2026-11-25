//! Validation of a rendered paper: structure first, then a real compile.
//!
//! Structural problems are collected together, because the repair model
//! can fix several in one pass. The compile runs only on a structurally
//! sound paper, in a scratch copy, with shell escape off and a time limit.
//! Without a TeX toolchain the report says the PDF was not built; that is
//! not a failure.

use crate::data::PaperData;
use crate::draft::{Draft, SECTIONS};
use crate::references::{bib_keys, Reference};
use crate::render::Assembled;
use crate::text::{parse_prose, visible_len};
use anyhow::{Context, Result};
use genesis_core::fsutil;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

/// The supplement must say something: this many visible characters.
const MIN_SUPPLEMENT_CHARS: usize = 100;
const COMPILE_STEP_TIMEOUT: Duration = Duration::from_secs(180);
const REQUIRED_FILES: &[&str] = &[
    "paper.tex",
    "paper.md",
    "sections.json",
    "claims_grounding.json",
    "paper_data.json",
    "references.bib",
    "references_provenance.json",
];

/// `paper/validation_report.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct ValidationReport {
    pub ok: bool,
    pub hard_errors: Vec<String>,
    pub soft_warnings: Vec<String>,
    pub missing_sections: Vec<String>,
    pub citation_count: usize,
    pub bibtex_key_count: usize,
    pub claim_count: usize,
    pub pdf_built: bool,
    /// Why no PDF was built although nothing is wrong with the paper.
    pub pdf_note: Option<String>,
    pub latex_log_tail: Option<String>,
}

/// The structural checks: files, sections, citations, the bibliography
/// against its provenance, claims against grounding ids, placeholders.
pub fn check_structure(
    dir: &Path,
    draft: &Draft,
    data: &PaperData,
    references: &[Reference],
    assembled: &Assembled,
    min_refs: usize,
) -> ValidationReport {
    let mut report = ValidationReport {
        claim_count: draft.claims.len(),
        citation_count: assembled.cite_keys.len(),
        ..Default::default()
    };
    let errors = &mut report.hard_errors;

    for name in REQUIRED_FILES {
        if !dir.join(name).is_file() {
            errors.push(format!("required file `{name}` is missing"));
        }
    }
    for spec in SECTIONS {
        if draft.section(spec.key).trim().is_empty() {
            report.missing_sections.push(spec.key.to_string());
            errors.push(format!("the {} section is missing or empty", spec.heading));
        }
    }
    if draft.title.trim().is_empty() {
        report
            .soft_warnings
            .push("the draft has no title; a default title was used".to_string());
    }
    let supplement = visible_len(&parse_prose(draft.section("supplementary")));
    if supplement > 0 && supplement < MIN_SUPPLEMENT_CHARS {
        errors.push(format!(
            "the Supplementary Information section has {supplement} visible characters; it needs at least {MIN_SUPPLEMENT_CHARS}"
        ));
    }

    let bibliography = std::fs::read_to_string(dir.join("references.bib")).unwrap_or_default();
    let keys = bib_keys(&bibliography);
    report.bibtex_key_count = keys.len();
    let key_set: HashSet<&str> = keys.iter().map(String::as_str).collect();
    if key_set.len() != keys.len() {
        errors.push("references.bib has duplicate entry keys".to_string());
    }
    let provenance: HashSet<&str> = references
        .iter()
        .map(|reference| reference.citation_key.as_str())
        .collect();
    let ids: HashSet<&str> = references
        .iter()
        .map(|reference| reference.id.as_str())
        .collect();
    if provenance.len() != references.len() || ids.len() != references.len() {
        errors.push("references_provenance.json has duplicate citation keys or ids".to_string());
    }
    for key in keys.iter().filter(|key| !provenance.contains(key.as_str())) {
        errors.push(format!(
            "bibliography entry `{key}` has no provenance record"
        ));
    }
    for key in references
        .iter()
        .map(|reference| reference.citation_key.as_str())
        .filter(|key| !key_set.contains(key))
    {
        errors.push(format!(
            "provenance record `{key}` has no bibliography entry"
        ));
    }
    for key in assembled
        .cite_keys
        .iter()
        .filter(|key| !key_set.contains(key.as_str()))
    {
        errors.push(format!(
            "the paper cites `{key}`, which is not a retrieved reference"
        ));
    }

    if draft.claims.is_empty() {
        errors.push("claims_grounding.json holds no claims".to_string());
    }
    for (index, claim) in draft.claims.iter().enumerate() {
        for id in claim
            .source_ids
            .iter()
            .filter(|id| !data.grounding_sources.contains_key(*id) && !ids.contains(id.as_str()))
        {
            errors.push(format!(
                "claim {} cites grounding id `{id}`, which does not exist",
                index + 1
            ));
        }
        for key in claim
            .citation_keys
            .iter()
            .filter(|key| !key_set.contains(key.as_str()))
        {
            errors.push(format!(
                "claim {} cites reference `{key}`, which was not retrieved",
                index + 1
            ));
        }
    }
    for token in &assembled.unresolved {
        errors.push(format!(
            "placeholder `{token}` names no table or figure this run has"
        ));
    }
    if !references.is_empty() && assembled.cite_keys.len() < min_refs.min(references.len()) {
        report.soft_warnings.push(format!(
            "the paper cites {} of the {} retrieved references; the target is at least {}",
            assembled.cite_keys.len(),
            references.len(),
            min_refs.min(references.len())
        ));
    }
    report.ok = report.hard_errors.is_empty();
    report
}

/// The first directory on `PATH` holding an executable named `tool`.
pub fn find_tool(tool: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(tool))
        .find(|candidate| candidate.is_file())
}

/// Whether a paper can be compiled on this machine.
pub fn tex_available() -> bool {
    find_tool("pdflatex").is_some() && find_tool("bibtex").is_some()
}

async fn run_step(program: &Path, args: &[&str], cwd: &Path) -> Result<(bool, String)> {
    let mut command = tokio::process::Command::new(program);
    command
        .args(args)
        .current_dir(cwd)
        .env("HOME", cwd)
        .env("openout_any", "p")
        .env("openin_any", "p")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let output = tokio::time::timeout(COMPILE_STEP_TIMEOUT, command.output())
        .await
        .with_context(|| {
            format!(
                "{} did not finish within {} seconds",
                program.display(),
                COMPILE_STEP_TIMEOUT.as_secs()
            )
        })?
        .with_context(|| format!("starting {}", program.display()))?;
    let mut log = String::from_utf8_lossy(&output.stdout).into_owned();
    log.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok((output.status.success(), log))
}

/// What a compile attempt produced.
#[derive(Debug, Clone, PartialEq)]
pub enum Compiled {
    Built,
    /// No toolchain: nothing was attempted.
    Skipped(String),
    /// The toolchain rejected the paper; the text is the end of its log.
    Failed(String),
}

/// Compile `paper.tex` in a scratch copy and bring `paper.pdf` back.
pub async fn compile_pdf(dir: &Path) -> Compiled {
    let (Some(pdflatex), Some(bibtex)) = (find_tool("pdflatex"), find_tool("bibtex")) else {
        return Compiled::Skipped(
            "pdflatex and bibtex are not both on PATH, so no PDF was built".to_string(),
        );
    };
    let build = dir.join(".latex_build");
    let _ = std::fs::remove_dir_all(&build);
    let outcome = async {
        std::fs::create_dir_all(&build)?;
        for name in ["paper.tex", "references.bib"] {
            std::fs::copy(dir.join(name), build.join(name))
                .with_context(|| format!("copying {name} into the build directory"))?;
        }
        let latex = [
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "paper.tex",
        ];
        let (ok, log) = run_step(&pdflatex, &latex, &build).await?;
        if !ok {
            return Ok(Compiled::Failed(fsutil::tail(&log, 3000)));
        }
        let cites = std::fs::read_to_string(build.join("paper.aux"))
            .is_ok_and(|aux| aux.contains("\\citation{"));
        if cites {
            let (ok, log) = run_step(&bibtex, &["paper"], &build).await?;
            if !ok {
                return Ok(Compiled::Failed(fsutil::tail(&log, 3000)));
            }
        }
        for _ in 0..2 {
            let (ok, log) = run_step(&pdflatex, &latex, &build).await?;
            if !ok {
                return Ok(Compiled::Failed(fsutil::tail(&log, 3000)));
            }
        }
        std::fs::copy(build.join("paper.pdf"), dir.join("paper.pdf"))
            .context("the compile produced no paper.pdf")?;
        anyhow::Ok(Compiled::Built)
    }
    .await;
    let _ = std::fs::remove_dir_all(&build);
    outcome.unwrap_or_else(|error| Compiled::Failed(format!("{error:#}")))
}

/// The line of a LaTeX log that names the error.
fn first_error(log: &str) -> String {
    log.lines()
        .find(|line| line.starts_with('!'))
        .unwrap_or("see latex_log_tail")
        .trim()
        .to_string()
}

/// Validate the paper in `dir` and write `validation_report.json`.
pub async fn validate(
    dir: &Path,
    draft: &Draft,
    data: &PaperData,
    references: &[Reference],
    assembled: &Assembled,
    min_refs: usize,
    compile: bool,
) -> Result<ValidationReport> {
    let mut report = check_structure(dir, draft, data, references, assembled, min_refs);
    let _ = std::fs::remove_file(dir.join("paper.pdf"));
    if !compile {
        report.pdf_note =
            Some("PDF compilation is turned off (paper.compile_pdf=false)".to_string());
    } else if report.ok {
        match compile_pdf(dir).await {
            Compiled::Built => report.pdf_built = true,
            Compiled::Skipped(note) => report.pdf_note = Some(note),
            Compiled::Failed(log) => {
                report
                    .hard_errors
                    .push(format!("LaTeX compilation failed: {}", first_error(&log)));
                report.latex_log_tail = Some(log);
                report.ok = false;
            }
        }
    } else {
        report.pdf_note = Some("not compiled: the structural errors come first".to_string());
    }
    fsutil::write_json(&dir.join("validation_report.json"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draft::Claim;
    use crate::render::assemble;

    fn complete_draft() -> Draft {
        let mut draft = Draft {
            title: "T".into(),
            ..Default::default()
        };
        for spec in SECTIONS {
            draft.sections.insert(spec.key.to_string(), format!("{} prose that is long enough to count as a real section of this paper, well past the minimum length that is required.", spec.heading));
        }
        draft.claims.push(Claim {
            claim: "c".into(),
            ..Default::default()
        });
        draft
    }

    fn write_all(dir: &Path, references: &[Reference]) {
        for name in REQUIRED_FILES {
            std::fs::write(dir.join(name), "{}").unwrap();
        }
        std::fs::write(
            dir.join("references.bib"),
            crate::references::write_bibtex(references),
        )
        .unwrap();
    }

    #[test]
    fn a_sound_paper_passes_the_structural_checks() {
        let dir = tempfile::tempdir().unwrap();
        let references = vec![Reference {
            id: "ref.s2.a".into(),
            citation_key: "smith2020".into(),
            title: "T".into(),
            ..Default::default()
        }];
        write_all(dir.path(), &references);
        let mut draft = complete_draft();
        draft
            .sections
            .insert("related_work".into(), "Prior work [@smith2020].".into());
        draft.claims[0].source_ids = vec!["ref.s2.a".into()];
        let data = PaperData::default();
        let report = check_structure(
            dir.path(),
            &draft,
            &data,
            &references,
            &assemble(&draft, &data, &references),
            8,
        );
        assert!(report.ok, "{:?}", report.hard_errors);
        assert_eq!(
            (
                report.citation_count,
                report.bibtex_key_count,
                report.claim_count
            ),
            (1, 1, 1)
        );
        assert!(report.soft_warnings.is_empty());
    }

    #[test]
    fn every_structural_problem_is_reported_together() {
        let dir = tempfile::tempdir().unwrap();
        let references = vec![Reference {
            id: "ref.s2.a".into(),
            citation_key: "smith2020".into(),
            ..Default::default()
        }];
        write_all(
            dir.path(),
            &[Reference {
                citation_key: "orphan1999".into(),
                ..Default::default()
            }],
        );
        std::fs::remove_file(dir.path().join("paper.md")).unwrap();
        let mut draft = complete_draft();
        draft.title.clear();
        draft.sections.insert("method".into(), " ".into());
        draft
            .sections
            .insert("supplementary".into(), "Too short.".into());
        draft.sections.insert(
            "results".into(),
            "See [@ghost2020].\n\n{{TABLE:ablations}}".into(),
        );
        draft.claims[0].source_ids = vec!["result.nope".into()];
        draft.claims[0].citation_keys = vec!["ghost2020".into()];
        let data = PaperData::default();
        let report = check_structure(
            dir.path(),
            &draft,
            &data,
            &references,
            &assemble(&draft, &data, &references),
            8,
        );
        assert!(!report.ok);
        let all = report.hard_errors.join("\n");
        for needle in [
            "required file `paper.md` is missing",
            "the Method section is missing or empty",
            "Supplementary Information section has 9 visible characters",
            "bibliography entry `orphan1999` has no provenance record",
            "provenance record `smith2020` has no bibliography entry",
            "the paper cites `ghost2020`",
            "claim 1 cites grounding id `result.nope`",
            "claim 1 cites reference `ghost2020`",
            "placeholder `{{TABLE:ablations}}`",
        ] {
            assert!(all.contains(needle), "missing `{needle}` in:\n{all}");
        }
        assert_eq!(report.missing_sections, ["method"]);
        assert_eq!(
            report.soft_warnings,
            ["the draft has no title; a default title was used"]
        );
        let empty = Draft::default();
        assert!(check_structure(
            dir.path(),
            &empty,
            &data,
            &[],
            &assemble(&empty, &data, &[]),
            0
        )
        .hard_errors
        .iter()
        .any(|e| e.contains("holds no claims")));
    }

    #[test]
    fn the_error_line_of_a_log_is_found() {
        assert_eq!(
            first_error("This is pdfTeX\n! Undefined control sequence.\nl.12 \\foo"),
            "! Undefined control sequence."
        );
        assert_eq!(first_error("nothing"), "see latex_log_tail");
        assert!(find_tool("definitely-not-a-real-tool-name").is_none());
    }
}
