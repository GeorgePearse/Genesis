// Circle packing: place 26 circles in the unit square, maximising the sum of radii.

const N: usize = 26;

// EVOLVE-BLOCK-START
/// Returns the circles as (x, y, r). This baseline puts equal circles on a
/// 6 x 5 grid and leaves most of the square unused.
fn construct_packing() -> Vec<(f64, f64, f64)> {
    let (columns, rows) = (6usize, 5usize);
    let radius = 0.07;
    let mut circles = Vec::with_capacity(N);
    for index in 0..N {
        let column = index % columns;
        let row = index / columns;
        let x = (column as f64 + 0.5) / columns as f64;
        let y = (row as f64 + 0.5) / rows as f64;
        circles.push((x, y, radius));
    }
    circles
}
// EVOLVE-BLOCK-END

fn main() {
    let circles = construct_packing();
    for (x, y, r) in circles.iter().take(N) {
        println!("{x:.12} {y:.12} {r:.12}");
    }
}
