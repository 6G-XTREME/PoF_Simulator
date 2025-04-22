use geo::{Point, Polygon, LineString, MultiPolygon};
use std::f64::consts::PI;
use geo_clipper::Clipper;

/// Calculate Apollonius circle path loss
pub fn apollonius_circle_path_loss(
    p1: (f64, f64),
    p2: (f64, f64),
    w1: f64,
    w2: f64,
    alpha: f64,
) -> Option<(f64, f64, f64)> {
    let lambda = (w1 / w2).powf(1.0 / alpha);

    if (1.0 - lambda * lambda).abs() < 1e-6 {
        return None;
    }

    let cx = (p1.0 - p2.0 * lambda * lambda) / (1.0 - lambda * lambda);
    let cy = (p1.1 - p2.1 * lambda * lambda) / (1.0 - lambda * lambda);
    let r = lambda * ((p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2)).sqrt().abs() / (1.0 - lambda * lambda).abs();

    Some((cx, cy, r))
}

/// Generate a circle polygon from a center and radius
pub fn create_circular_region(center: (f64, f64), radius: f64, steps: usize) -> Polygon {
    let (cx, cy) = center;
    let theta = 2.0 * PI / steps as f64;
    let coords: Vec<_> = (0..=steps)
        .map(|i| {
            let angle = i as f64 * theta;
            (cx + radius * angle.cos(), cy + radius * angle.sin())
        })
        .collect();
    Polygon::new(LineString::from(coords), vec![])
}

/// Create a base region for a base station
pub fn create_base_region_for_bs(
    bs_position: (f64, f64),
    max_radius_km: f64,
    euclidean_to_km_scale: f64,
) -> Polygon {
    create_circular_region(bs_position, max_radius_km * euclidean_to_km_scale, 50)
}

/// Get the Euclidean distance between two points
pub fn get_euclidean_distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    ((p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2)).sqrt()
}

/// Get the distance in kilometers between two points
pub fn get_distance_in_kilometers(p1: (f64, f64), p2: (f64, f64), euclidean_to_km_scale: f64) -> f64 {
    get_euclidean_distance(p1, p2) * euclidean_to_km_scale
}

/// Calculate the perpendicular bisector between two points
pub fn perpendicular_bisector(p1: (f64, f64), p2: (f64, f64)) -> (f64, f64) {
    let x_med = (p1.0 + p2.0) / 2.0;
    let y_med = (p1.1 + p2.1) / 2.0;

    let a = -1.0 / ((p2.1 - p1.1) / (p2.0 - p1.0));
    let b = y_med - (a * x_med);

    (b, a + b)
}

/// Get the dominance area between two points
pub fn get_dominance_area(
    p1: (f64, f64),
    p2: (f64, f64),
    polygon_bounds: &[(f64, f64)],
) -> Polygon {
    let (b, a_plus_b) = perpendicular_bisector(p1, p2);
    let whole_region = Polygon::new(LineString::from(polygon_bounds.to_vec()), vec![]);
    
    // TODO: Implement polygon clipping similar to Python's polyclip
    // For now, we'll return a simplified version
    let point = Point::new(p1.0, p1.1);
    let mut polygon = whole_region.clone();
    
    if !polygon.contains(&point) {
        polygon = polygon.difference(&whole_region, 1.0).into_iter().next().unwrap();
    }
    
    polygon
}

/// Search for the closest base station to a point
pub fn search_closest_bs(point: (f64, f64), regions: &[Polygon]) -> usize {
    let point = Point::new(point.0, point.1);
    let mut closest = 0;

    for (idx, region) in regions.iter().enumerate() {
        if region.contains(&point) {
            closest = idx;
        }
    }

    closest
}