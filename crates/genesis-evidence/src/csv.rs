//! Reading and writing the aggregated tables as CSV (RFC 4180).
//!
//! A cell is quoted when it holds a comma, a quote or a line break, and
//! every record ends with a line feed, so any string survives a round trip.

use anyhow::Result;
use genesis_core::fsutil;
use std::path::Path;

fn encode_cell(cell: &str) -> String {
    if cell.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", cell.replace('"', "\"\""))
    } else {
        cell.to_string()
    }
}

/// One record, terminated by a line feed. A record has at least one cell.
pub fn encode_row(cells: &[String]) -> String {
    let mut line: String = cells
        .iter()
        .map(|cell| encode_cell(cell))
        .collect::<Vec<_>>()
        .join(",");
    line.push('\n');
    line
}

pub fn encode(header: &[&str], rows: &[Vec<String>]) -> String {
    let header: Vec<String> = header.iter().map(|name| name.to_string()).collect();
    let mut text = encode_row(&header);
    for row in rows {
        text.push_str(&encode_row(row));
    }
    text
}

/// Every record of `text`, header included. A final record without a line
/// feed is kept.
pub fn parse(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut cell = String::new();
    let mut in_quotes = false;
    let mut quoted = false;
    let mut pending = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        pending = true;
        if in_quotes {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    chars.next();
                    cell.push('"');
                }
                '"' => in_quotes = false,
                other => cell.push(other),
            }
            continue;
        }
        match c {
            '"' if cell.is_empty() && !quoted => {
                in_quotes = true;
                quoted = true;
            }
            ',' => {
                row.push(std::mem::take(&mut cell));
                quoted = false;
            }
            '\r' if chars.peek() == Some(&'\n') => {}
            '\n' => {
                row.push(std::mem::take(&mut cell));
                rows.push(std::mem::take(&mut row));
                quoted = false;
                pending = false;
            }
            other => cell.push(other),
        }
    }
    if pending {
        row.push(cell);
        rows.push(row);
    }
    rows
}

pub fn write(path: &Path, header: &[&str], rows: &[Vec<String>]) -> Result<()> {
    fsutil::write_text(path, &encode(header, rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(cells: &[&str]) -> Vec<String> {
        cells.iter().map(|cell| cell.to_string()).collect()
    }

    #[test]
    fn awkward_cells_are_quoted_and_read_back() {
        let rows = vec![
            row(&[
                "plain",
                "has,comma",
                "has \"quotes\"",
                "two\nlines",
                "cr\rhere",
                "",
            ]),
            row(&[""]),
        ];
        let text = encode(&["a", "b", "c", "d", "e", "f"], &rows);
        assert!(text.starts_with("a,b,c,d,e,f\nplain,\"has,comma\",\"has \"\"quotes\"\"\",\"two\nlines\",\"cr\rhere\",\n"));
        let parsed = parse(&text);
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[1], rows[0]);
        assert_eq!(parsed[2], rows[1]);
    }

    #[test]
    fn foreign_line_endings_and_a_missing_final_newline_are_accepted() {
        assert_eq!(
            parse("a,b\r\n1,2"),
            vec![row(&["a", "b"]), row(&["1", "2"])]
        );
        assert!(parse("").is_empty());
        assert_eq!(parse("\n"), vec![row(&[""])]);
    }
}
