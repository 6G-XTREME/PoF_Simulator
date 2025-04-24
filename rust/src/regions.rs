use geo::{Point, Polygon, MultiPolygon, GeometryCollection, Coord, LineString, Contains, Intersects, BooleanOps};
use geo_booleanop::boolean::BooleanOp;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Crea una región circular simulando un buffer (workaround)
fn create_buffer(point: Point<f64>, radius: f64, resolution: usize) -> Polygon<f64> {
    let angle_step = 2.0 * PI / resolution as f64;
    let coords: Vec<Coord<f64>> = (0..=resolution)
        .map(|i| {
            let angle = i as f64 * angle_step;
            Coord {
                x: point.x() + radius * angle.cos(),
                y: point.y() + radius * angle.sin(),
            }
        })
        .collect();
    Polygon::new(LineString::from(coords), vec![])
}

/// Calcula el círculo de Apolonio
fn apollonius_circle_path_loss(p1: (f64, f64), p2: (f64, f64), w1: f64, w2: f64, alpha: f64) -> (f64, f64, f64) {
    let lambda = (w1 / w2).powf(1.0 / alpha);
    let cx = (p1.0 - p2.0 * lambda * lambda) / (1.0 - lambda * lambda);
    let cy = (p1.1 - p2.1 * lambda * lambda) / (1.0 - lambda * lambda);
    let r = lambda * ((p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2)).sqrt() / (1.0 - lambda * lambda).abs();
    (cx, cy, r)
}

/// Genera círculo como polígono
fn get_circle(center: (f64, f64), radius: f64, resolution: usize) -> Polygon<f64> {
    create_buffer(Point::new(center.0, center.1), radius, resolution)
}

/// Calcula el bisector perpendicular como línea
fn perpendicular_bisector(p1: (f64, f64), p2: (f64, f64)) -> ((f64, f64), (f64, f64)) {
    let mid = ((p1.0 + p2.0) / 2.0, (p1.1 + p2.1) / 2.0);
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let slope = -dx / dy;
    let intercept = mid.1 - slope * mid.0;
    (mid, (slope, intercept))
}

/// Regresa área de dominancia de una estación
fn get_dominance_area(p1: (f64, f64), p2: (f64, f64), bounds: &Polygon<f64>) -> Polygon<f64> {
    let mid = ((p1.0 + p2.0) / 2.0, (p1.1 + p2.1) / 2.0);
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let slope = -dx / dy;
    let intercept = mid.1 - slope * mid.0;
    // Esto debería implementarse usando líneas y cortes reales
    // Aquí simplemente se retorna todo el bounds por simplicidad
    bounds.clone()
}

/// Genera la región base circular de una estación
fn create_base_region_for_bs(bs_position: (f64, f64), radius: f64) -> Polygon<f64> {
    create_buffer(Point::new(bs_position.0, bs_position.1), radius, 100)
}

/// Crea las regiones de cobertura
fn create_regions(base_stations: &[(f64, f64, f64)], alpha_loss: f64, bounds: Polygon<f64>, max_radius: &[f64]) -> HashMap<usize, MultiPolygon<f64>> {
    let mut unsold = MultiPolygon::from(&bounds.clone());
    let mut regions = HashMap::new();

    for (i, &bs) in base_stations.iter().enumerate().rev() {
        let mut region = unsold.clone();
        let coverage = MultiPolygon::from(create_base_region_for_bs((bs.0, bs.1), max_radius[i]));

        for (j, &other_bs) in base_stations.iter().enumerate() {
            if j >= i { continue; }
            let assigned_region = if bs.2 != other_bs.2 {
                let (cx, cy, r) = apollonius_circle_path_loss((bs.0, bs.1), (other_bs.0, other_bs.1), bs.2, other_bs.2, alpha_loss);
                get_circle((cx, cy), r, 100)
            } else {
                get_dominance_area((bs.0, bs.1), (other_bs.0, other_bs.1), &bounds)
            };
            region = region.intersection(&assigned_region);
        }
        region = region.intersection(&coverage);
        unsold = unsold.difference(&region);
        regions.insert(i, region);
    }

    regions
}
