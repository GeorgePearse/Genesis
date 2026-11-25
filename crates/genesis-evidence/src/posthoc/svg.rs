//! Small self-contained SVG charts for the post-hoc analyses: black axes on
//! white, so they print, with no script and no external reference.

const WIDTH: f64 = 720.0;
const HEIGHT: f64 = 420.0;
const LEFT: f64 = 72.0;
const RIGHT: f64 = 24.0;
const TOP: f64 = 44.0;
const BOTTOM: f64 = 64.0;
const COLOURS: [&str; 6] = [
    "#1f4e79", "#c0392b", "#2e7d32", "#7b1fa2", "#ef6c00", "#00838f",
];

/// A named sequence of points.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    pub name: String,
    pub points: Vec<(f64, f64)>,
    /// Join the points with a line; otherwise draw them as dots.
    pub line: bool,
}

fn escape(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '&' => "&amp;".to_string(),
            '<' => "&lt;".to_string(),
            '>' => "&gt;".to_string(),
            '"' => "&quot;".to_string(),
            '\'' => "&apos;".to_string(),
            c if c.is_control() => " ".to_string(),
            c => c.to_string(),
        })
        .collect()
}

fn tick_label(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    if value.abs() >= 100_000.0 || value.abs() < 0.001 {
        return format!("{value:.2e}");
    }
    let text = format!("{value:.3}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// The range to draw `values` in: padded, and never empty.
fn bounds(values: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for value in values.filter(|value| value.is_finite()) {
        low = low.min(value);
        high = high.max(value);
    }
    if low > high {
        return None;
    }
    let span = high - low;
    let pad = if span > 0.0 {
        span * 0.06
    } else {
        low.abs().max(1.0) * 0.1
    };
    let (low, high) = (low - pad, high + pad);
    // Near the limits of a float the padded range overflows; nothing can be
    // placed on such an axis.
    (low.is_finite() && high.is_finite() && (high - low).is_finite()).then_some((low, high))
}

fn frame(title: &str, x_label: &str, y_label: &str, body: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {WIDTH} {HEIGHT}\" width=\"{WIDTH}\" height=\"{HEIGHT}\" \
         font-family=\"Helvetica, Arial, sans-serif\" font-size=\"12\">\n\
         <rect width=\"{WIDTH}\" height=\"{HEIGHT}\" fill=\"#ffffff\"/>\n\
         <text x=\"{LEFT}\" y=\"24\" font-size=\"15\" font-weight=\"bold\" fill=\"#000000\">{}</text>\n\
         {body}\
         <text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"#000000\">{}</text>\n\
         <text x=\"16\" y=\"{}\" text-anchor=\"middle\" fill=\"#000000\" transform=\"rotate(-90 16 {})\">{}</text>\n\
         </svg>\n",
        escape(title),
        (LEFT + WIDTH - RIGHT) / 2.0,
        HEIGHT - 12.0,
        escape(x_label),
        (TOP + HEIGHT - BOTTOM) / 2.0,
        (TOP + HEIGHT - BOTTOM) / 2.0,
        escape(y_label),
    )
}

fn empty(title: &str, x_label: &str, y_label: &str) -> String {
    let body = format!(
        "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"#555555\">no data to plot</text>\n",
        WIDTH / 2.0,
        HEIGHT / 2.0
    );
    frame(title, x_label, y_label, &body)
}

struct Scale {
    low: f64,
    high: f64,
    start: f64,
    end: f64,
}

impl Scale {
    fn at(&self, value: f64) -> f64 {
        self.start + (value - self.low) / (self.high - self.low) * (self.end - self.start)
    }
}

fn axes(x: Option<&Scale>, y: &Scale) -> String {
    let (x0, x1, y0, y1) = (LEFT, WIDTH - RIGHT, HEIGHT - BOTTOM, TOP);
    let mut out = String::new();
    for step in 0..=4 {
        let fraction = f64::from(step) / 4.0;
        let value = y.low + (y.high - y.low) * fraction;
        let position = y.at(value);
        out.push_str(&format!(
            "<line x1=\"{x0}\" y1=\"{position:.1}\" x2=\"{x1}\" y2=\"{position:.1}\" stroke=\"#dddddd\"/>\n\
             <text x=\"{}\" y=\"{:.1}\" text-anchor=\"end\" fill=\"#000000\">{}</text>\n",
            x0 - 6.0,
            position + 4.0,
            tick_label(value)
        ));
        if let Some(x) = x {
            let value = x.low + (x.high - x.low) * fraction;
            out.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{}\" text-anchor=\"middle\" fill=\"#000000\">{}</text>\n",
                x.at(value),
                y0 + 18.0,
                tick_label(value)
            ));
        }
    }
    out.push_str(&format!(
        "<line x1=\"{x0}\" y1=\"{y0}\" x2=\"{x1}\" y2=\"{y0}\" stroke=\"#000000\"/>\n\
         <line x1=\"{x0}\" y1=\"{y0}\" x2=\"{x0}\" y2=\"{y1}\" stroke=\"#000000\"/>\n"
    ));
    out
}

/// Lines and dots over two numeric axes, with a legend.
pub fn xy_chart(title: &str, x_label: &str, y_label: &str, series: &[Series]) -> String {
    let finite = |point: &&(f64, f64)| point.0.is_finite() && point.1.is_finite();
    let points = || {
        series
            .iter()
            .flat_map(|series| series.points.iter())
            .filter(finite)
    };
    let (Some(x_bounds), Some(y_bounds)) =
        (bounds(points().map(|p| p.0)), bounds(points().map(|p| p.1)))
    else {
        return empty(title, x_label, y_label);
    };
    let x = Scale {
        low: x_bounds.0,
        high: x_bounds.1,
        start: LEFT,
        end: WIDTH - RIGHT,
    };
    let y = Scale {
        low: y_bounds.0,
        high: y_bounds.1,
        start: HEIGHT - BOTTOM,
        end: TOP,
    };
    let mut body = axes(Some(&x), &y);
    for (index, series) in series.iter().enumerate() {
        let colour = COLOURS[index % COLOURS.len()];
        let placed: Vec<(f64, f64)> = series
            .points
            .iter()
            .filter(finite)
            .map(|p| (x.at(p.0), y.at(p.1)))
            .collect();
        if series.line && placed.len() > 1 {
            let path: Vec<String> = placed
                .iter()
                .map(|(px, py)| format!("{px:.1},{py:.1}"))
                .collect();
            body.push_str(&format!(
                "<polyline points=\"{}\" fill=\"none\" stroke=\"{colour}\" stroke-width=\"2\"/>\n",
                path.join(" ")
            ));
        }
        if !series.line || placed.len() <= 60 {
            for (px, py) in &placed {
                body.push_str(&format!(
                    "<circle cx=\"{px:.1}\" cy=\"{py:.1}\" r=\"3\" fill=\"{colour}\"/>\n"
                ));
            }
        }
        let legend_y = TOP + 14.0 * index as f64;
        body.push_str(&format!(
            "<rect x=\"{}\" y=\"{:.1}\" width=\"10\" height=\"10\" fill=\"{colour}\"/>\n\
             <text x=\"{}\" y=\"{:.1}\" fill=\"#000000\">{}</text>\n",
            WIDTH - RIGHT - 170.0,
            legend_y - 9.0,
            WIDTH - RIGHT - 155.0,
            legend_y,
            escape(&series.name)
        ));
    }
    frame(title, x_label, y_label, &body)
}

/// One bar per label, from zero.
pub fn bar_chart(title: &str, y_label: &str, bars: &[(String, f64)]) -> String {
    let bars: Vec<&(String, f64)> = bars.iter().filter(|bar| bar.1.is_finite()).collect();
    let Some((low, high)) = bounds(bars.iter().map(|bar| bar.1).chain(std::iter::once(0.0))) else {
        return empty(title, "", y_label);
    };
    if bars.is_empty() {
        return empty(title, "", y_label);
    }
    let y = Scale {
        low: low.min(0.0),
        high: high.max(0.0),
        start: HEIGHT - BOTTOM,
        end: TOP,
    };
    let mut body = axes(None, &y);
    let slot = (WIDTH - LEFT - RIGHT) / bars.len() as f64;
    let zero = y.at(0.0);
    for (index, (label, value)) in bars.iter().enumerate() {
        let top = y.at(*value);
        let left = LEFT + slot * index as f64 + slot * 0.15;
        body.push_str(&format!(
            "<rect x=\"{left:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{}\"/>\n\
             <text x=\"{:.1}\" y=\"{}\" text-anchor=\"middle\" fill=\"#000000\">{}</text>\n",
            top.min(zero),
            slot * 0.7,
            (top - zero).abs().max(0.5),
            COLOURS[0],
            left + slot * 0.35,
            HEIGHT - BOTTOM + 18.0,
            escape(&genesis_core::fsutil::truncate(
                label,
                (slot / 7.0).max(4.0) as usize
            ))
        ));
    }
    frame(title, "", y_label, &body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charts_are_well_formed_and_escape_their_labels() {
        let series = vec![
            Series {
                name: "best <so far>".into(),
                points: vec![(0.0, 1.0), (1.0, 2.5), (2.0, 2.5)],
                line: true,
            },
            Series {
                name: "evaluated".into(),
                points: vec![(0.0, 1.0), (1.0, f64::NAN), (2.0, 0.5)],
                line: false,
            },
        ];
        let chart = xy_chart("Score & generation", "generation", "score", &series);
        assert!(chart.starts_with("<svg xmlns=") && chart.trim_end().ends_with("</svg>"));
        assert!(chart.contains("Score &amp; generation") && chart.contains("best &lt;so far&gt;"));
        assert_eq!(chart.matches("<polyline").count(), 1);
        assert_eq!(
            chart.matches("<circle").count(),
            5,
            "three line points and two finite dots"
        );
        assert!(!chart.contains("NaN") && !chart.contains("inf"));

        let bars = bar_chart(
            "Share improving",
            "share",
            &[
                ("diff".into(), 0.25),
                ("full".into(), -0.5),
                ("bad".into(), f64::INFINITY),
            ],
        );
        assert_eq!(
            bars.matches("<rect").count(),
            3,
            "the background and two finite bars"
        );
    }

    /// Found by `charts_draw_any_numbers_without_a_non_finite_coordinate`
    /// (EPHYRA_REPLAY=17170835532124312781:5): one point at the largest
    /// float made the padded range overflow, and every coordinate became NaN.
    #[test]
    fn a_range_that_overflows_when_padded_is_not_plotted() {
        let lone = xy_chart(
            "t",
            "x",
            "y",
            &[Series {
                name: "s".into(),
                points: vec![(0.0, f64::MAX)],
                line: true,
            }],
        );
        assert!(!lone.contains("NaN") && !lone.contains("inf"), "{lone}");
        assert!(lone.contains("no data to plot"));
        let wide = xy_chart(
            "t",
            "x",
            "y",
            &[Series {
                name: "s".into(),
                points: vec![(0.0, -9e307), (1.0, 9e307)],
                line: true,
            }],
        );
        assert!(!wide.contains("NaN") && !wide.contains("inf"), "{wide}");
        let bars = bar_chart("t", "y", &[("a".into(), f64::MAX), ("b".into(), f64::MIN)]);
        assert!(!bars.contains("NaN") && !bars.contains("inf"), "{bars}");
    }

    #[test]
    fn degenerate_data_still_draws() {
        assert!(xy_chart("t", "x", "y", &[]).contains("no data to plot"));
        let flat = xy_chart(
            "t",
            "x",
            "y",
            &[Series {
                name: "one".into(),
                points: vec![(3.0, 3.0)],
                line: true,
            }],
        );
        assert!(flat.contains("<circle") && !flat.contains("NaN"));
        assert!(bar_chart("t", "y", &[]).contains("no data to plot"));
        let huge = xy_chart(
            "t",
            "x",
            "y",
            &[Series {
                name: "s".into(),
                points: vec![(0.0, -1e308), (1.0, 1e308)],
                line: true,
            }],
        );
        assert!(
            huge.contains("no data to plot"),
            "a span that overflows is not plotted"
        );
        assert_eq!(tick_label(0.5), "0.5");
        assert_eq!(tick_label(1234.0), "1234");
        assert_eq!(tick_label(1e9), "1.00e9");
    }
}
