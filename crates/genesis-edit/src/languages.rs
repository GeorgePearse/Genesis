//! The languages the patch layer knows: file extension, how an EVOLVE-BLOCK
//! marker is commented, and which code-fence tags a model may use.

/// One supported language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LanguageSpec {
    pub name: &'static str,
    pub extension: &'static str,
    /// What opens the comment that carries an EVOLVE-BLOCK marker.
    pub comment_prefix: &'static str,
    /// What closes it, for languages whose comments must be closed.
    pub comment_suffix: &'static str,
    /// Accepted code-fence tags, the canonical one first.
    pub fence_tags: &'static [&'static str],
    aliases: &'static [&'static str],
}

impl LanguageSpec {
    /// True when a comment can span lines, so a marker must open and close
    /// its comment on its own line.
    pub fn has_block_comments(&self) -> bool {
        !self.comment_suffix.is_empty()
    }

    /// The canonical marker line, such as `// EVOLVE-BLOCK-START`.
    pub fn marker_line(&self, marker: &str) -> String {
        if self.comment_suffix.is_empty() {
            format!("{} {marker}", self.comment_prefix)
        } else {
            format!("{} {marker} {}", self.comment_prefix, self.comment_suffix)
        }
    }
}

const fn line_comment(
    name: &'static str,
    extension: &'static str,
    comment_prefix: &'static str,
    fence_tags: &'static [&'static str],
    aliases: &'static [&'static str],
) -> LanguageSpec {
    LanguageSpec {
        name,
        extension,
        comment_prefix,
        comment_suffix: "",
        fence_tags,
        aliases,
    }
}

/// The ShinkaEvolve table, followed by the languages Genesis adds
/// (`javascript`, `typescript`, `c`, `bash`).
pub const LANGUAGES: &[LanguageSpec] = &[
    line_comment("python", "py", "#", &["python", "py"], &["py", "python3"]),
    line_comment("julia", "jl", "#", &["julia", "jl"], &["jl"]),
    line_comment(
        "cpp",
        "cpp",
        "//",
        &["cpp", "c++", "cc", "cxx"],
        &["c++", "cxx", "cc"],
    ),
    line_comment("cuda", "cu", "//", &["cuda", "cu"], &["cu"]),
    line_comment("go", "go", "//", &["go", "golang"], &["golang"]),
    line_comment("rust", "rs", "//", &["rust", "rs"], &["rs"]),
    line_comment("swift", "swift", "//", &["swift"], &[]),
    line_comment("json", "json", "//", &["json"], &[]),
    line_comment("json5", "json", "//", &["json5", "json"], &[]),
    line_comment(
        "verilog",
        "sv",
        "//",
        &["verilog", "systemverilog", "sv"],
        &["sv", "sverilog", "systemverilog"],
    ),
    line_comment(
        "fortran",
        "f90",
        "!",
        &["fortran", "f90", "f95", "f03", "f08"],
        &["f90", "f95", "f03", "f08"],
    ),
    LanguageSpec {
        name: "markdown",
        extension: "md",
        comment_prefix: "<!--",
        comment_suffix: "-->",
        fence_tags: &["markdown", "md"],
        aliases: &["md"],
    },
    LanguageSpec {
        name: "wolfram",
        extension: "wl",
        comment_prefix: "(*",
        comment_suffix: "*)",
        fence_tags: &["wolfram", "mathematica", "wl", "wls"],
        aliases: &["wl", "wls", "mathematica"],
    },
    line_comment(
        "javascript",
        "js",
        "//",
        &["javascript", "js", "jsx"],
        &["js", "jsx", "node"],
    ),
    line_comment(
        "typescript",
        "ts",
        "//",
        &["typescript", "ts", "tsx"],
        &["ts", "tsx"],
    ),
    line_comment("c", "c", "//", &["c"], &[]),
    line_comment(
        "bash",
        "sh",
        "#",
        &["bash", "sh", "shell"],
        &["sh", "shell"],
    ),
];

/// Look a language up by canonical name or alias, ignoring case and
/// surrounding whitespace.
pub fn language(name: &str) -> Option<&'static LanguageSpec> {
    let wanted = name.trim().to_lowercase();
    LANGUAGES
        .iter()
        .find(|spec| spec.name == wanted)
        .or_else(|| {
            LANGUAGES
                .iter()
                .find(|spec| spec.aliases.contains(&wanted.as_str()))
        })
}

/// Fence tags to try for `language`: the spelling asked for first, then the
/// language's own tags, then its other aliases (so a `python3` fence is
/// recognised). An unknown language is its own only tag.
pub fn fence_tags(language_name: &str) -> Vec<String> {
    let requested = language_name.trim().to_lowercase();
    let mut tags = vec![requested];
    if let Some(spec) = language(language_name) {
        for tag in spec.fence_tags.iter().chain(spec.aliases) {
            if !tags.iter().any(|known| known == tag) {
                tags.push((*tag).to_string());
            }
        }
    }
    tags
}

/// File extension without the dot; `txt` for an unknown language.
pub fn extension(language_name: &str) -> &'static str {
    language(language_name).map_or("txt", |spec| spec.extension)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefix(name: &str) -> &'static str {
        language(name).unwrap().comment_prefix
    }

    #[test]
    fn language_helpers_support_wolfram_aliases() {
        for alias in ["wolfram", "wl", "wls", "mathematica", "Mathematica"] {
            assert_eq!(language(alias).unwrap().name, "wolfram");
        }
        assert_eq!(
            (
                extension("wolfram"),
                extension("wl"),
                extension("mathematica")
            ),
            ("wl", "wl", "wl")
        );
        assert_eq!((prefix("wolfram"), prefix("wl")), ("(*", "(*"));
        let fences = fence_tags("wl");
        assert_eq!(fences[0], "wl");
        for tag in ["wolfram", "mathematica", "wls"] {
            assert!(fences.iter().any(|f| f == tag));
        }
        let fences = fence_tags("mathematica");
        assert_eq!(fences[0], "mathematica");
        assert!(fences.iter().any(|f| f == "wolfram"));
    }

    #[test]
    fn language_helpers_support_markdown_and_md_alias() {
        assert_eq!((extension("markdown"), prefix("markdown")), ("md", "<!--"));
        let fences = fence_tags("markdown");
        assert_eq!(fences[0], "markdown");
        assert!(fences.iter().any(|f| f == "md"));
        assert_eq!(language("md").unwrap().name, "markdown");
        assert_eq!((extension("md"), prefix("md")), ("md", "<!--"));
        let fences = fence_tags("md");
        assert_eq!(fences[0], "md");
        assert!(fences.iter().any(|f| f == "markdown"));
    }

    #[test]
    fn language_helpers_support_fortran_aliases() {
        for alias in ["f90", "f95", "f03", "f08"] {
            assert_eq!(language(alias).unwrap().name, "fortran");
            assert_eq!((extension(alias), prefix(alias)), ("f90", "!"));
        }
        assert_eq!((extension("fortran"), prefix("fortran")), ("f90", "!"));
        let fences = fence_tags("f95");
        assert_eq!(fences[0], "f95");
        for tag in ["fortran", "f90", "f03", "f08"] {
            assert!(fences.iter().any(|f| f == tag));
        }
    }

    #[test]
    fn language_helpers_support_go_julia_and_verilog_aliases() {
        for alias in ["go", "golang", "Go"] {
            assert_eq!(language(alias).unwrap().name, "go");
        }
        assert_eq!(
            (extension("go"), extension("golang"), prefix("golang")),
            ("go", "go", "//")
        );
        assert_eq!(fence_tags("golang"), ["golang", "go"]);
        assert_eq!(language("jl").unwrap().name, "julia");
        assert_eq!((extension("julia"), prefix("jl")), ("jl", "#"));
        assert_eq!(fence_tags("jl"), ["jl", "julia"]);
        for alias in ["sv", "sverilog", "systemverilog"] {
            assert_eq!(language(alias).unwrap().name, "verilog");
        }
        assert_eq!((extension("verilog"), prefix("sv")), ("sv", "//"));
        assert_eq!(
            fence_tags("sv"),
            ["sv", "verilog", "systemverilog", "sverilog"]
        );
    }

    #[test]
    fn cpp_aliases_are_all_accepted_fence_tags() {
        let mut tags = fence_tags("c++");
        tags.sort();
        assert_eq!(tags, ["c++", "cc", "cpp", "cxx"]);
    }

    #[test]
    fn genesis_additions_and_unknown_languages() {
        assert_eq!(
            (
                extension("javascript"),
                extension("ts"),
                extension("c"),
                extension("shell")
            ),
            ("js", "ts", "c", "sh")
        );
        assert_eq!((prefix("typescript"), prefix("bash")), ("//", "#"));
        assert_eq!(language("rs").unwrap().name, "rust");
        assert!(language("klingon").is_none());
        assert_eq!(extension("klingon"), "txt");
        assert_eq!(fence_tags(" Klingon "), ["klingon"]);
        assert_eq!(
            language("wolfram").unwrap().marker_line("EVOLVE-BLOCK-END"),
            "(* EVOLVE-BLOCK-END *)"
        );
        assert_eq!(
            language("rust").unwrap().marker_line("EVOLVE-BLOCK-START"),
            "// EVOLVE-BLOCK-START"
        );
    }
}
