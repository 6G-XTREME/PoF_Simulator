use geo::{Polygon, MultiPolygon};
use geo_clipper::Clipper;

pub fn intersect(a: &Polygon<f64>, b: &Polygon<f64>) -> MultiPolygon<f64> {
    a.intersection(b, 1.0)
}

pub fn subtract(a: &Polygon<f64>, b: &Polygon<f64>) -> MultiPolygon<f64> {
    a.difference(b, 1.0)
}
