use std::default::Default;

use bezierflattener::CBezierFlattener;

use crate::{bezierflattener::{CFlatteningSink, GpPointR, HRESULT, S_OK, CBezier}, tri_rasterize::rasterize_to_mask};

mod bezierflattener;
mod tri_rasterize;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Winding {
    EvenOdd,
    NonZero,
}

#[derive(Clone, Copy, Debug)]
pub enum PathOp {
    MoveTo(Point),
    LineTo(Point),
    QuadTo(Point, Point),
    CubicTo(Point, Point, Point),
    Close,
}

impl PathOp {
    fn transform(self, xform: &Transform) -> PathOp {
        match self {
            PathOp::MoveTo(p) => PathOp::MoveTo(xform.transform_point(p)),
            PathOp::LineTo(p) => PathOp::LineTo(xform.transform_point(p)),
            PathOp::QuadTo(p1, p2) => PathOp::QuadTo(
                xform.transform_point(p1),
                xform.transform_point(p2)
            ),
            PathOp::CubicTo(p1, p2, p3) => PathOp::CubicTo(
                xform.transform_point(p1),
                xform.transform_point(p2),
                xform.transform_point(p3),
            ),
            PathOp::Close => PathOp::Close,
        }
    }
}

/// Represents a complete path usable for filling or stroking.
#[derive(Clone, Debug)]
pub struct Path {
    pub ops: Vec<PathOp>,
    pub winding: Winding,
}

pub type Point = euclid::default::Point2D<f32>;
pub type Transform = euclid::default::Transform2D<f32>;
pub type Vector = euclid::default::Vector2D<f32>;



#[derive(Clone, PartialEq, Debug)]
pub struct StrokeStyle {
    pub width: f32,
    pub cap: LineCap,
    pub join: LineJoin,
    pub miter_limit: f32,
    pub dash_array: Vec<f32>,
    pub dash_offset: f32,
}

impl Default for StrokeStyle {
    fn default() -> Self {
        StrokeStyle {
            width: 1.,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            miter_limit: 10.,
            dash_array: Vec::new(),
            dash_offset: 0.,
        }
    }
}
#[derive(Debug)]
pub struct Vertex {
    x: f32,
    y: f32,
    coverage: f32
}

/// A helper struct used for constructing a `Path`.
pub struct PathBuilder {
    path: Path,
    vertices: Vec<Vertex>,
    transform: Transform
}



impl PathBuilder {
    pub fn new() -> PathBuilder {
        PathBuilder {
            path: Path {
                ops: Vec::new(),
                winding: Winding::NonZero,
            },
            vertices: Vec::new(),
            transform: Default::default()
        }
    }

    pub fn push_tri(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) {
        self.vertices.push(Vertex { x: x1, y: y1, coverage: 1.});
        self.vertices.push(Vertex { x: x2, y: y2, coverage: 1.});
        self.vertices.push(Vertex { x: x3, y: y3, coverage: 1.});

    }

    pub fn quad(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32, x4: f32, y4: f32) {
        self.move_to(x1, y1);
        self.line_to(x2, y2);
        self.line_to(x3, y3);
        self.line_to(x4, y4);
        self.close();

        self.push_tri(x1, y1, x2, y2, x3, y3);
        self.push_tri(x3, y3, x4, y4, x1, y1);
    }

    pub fn ramp(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32, x4: f32, y4: f32) {
        self.vertices.push(Vertex { x: x1, y: y1, coverage: 1.});
        self.vertices.push(Vertex { x: x2, y: y2, coverage: 0.});
        self.vertices.push(Vertex { x: x3, y: y3, coverage: 0.});

        self.vertices.push(Vertex { x: x3, y: y3, coverage: 0.});
        self.vertices.push(Vertex { x: x4, y: y4, coverage: 1.});
        self.vertices.push(Vertex { x: x1, y: y1, coverage: 1.});
    }

    // first edge is outside
    pub fn tri(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x3: f32, y3: f32) {
        self.move_to(x1, y1);
        self.line_to(x2, y2);
        self.line_to(x3, y3);
        self.close();

        self.push_tri(x1, y1, x2, y2, x3, y3);
    }

    /// Moves the current point to `x`, `y`
    pub fn move_to(&mut self, x: f32, y: f32) {
        self.path.ops.push(PathOp::MoveTo(Point::new(x, y)))
    }

    /// Adds a line segment from the current point to `x`, `y`
    pub fn line_to(&mut self, x: f32, y: f32) {
        self.path.ops.push(PathOp::LineTo(Point::new(x, y)))
    }

    /// Adds a quadratic bezier from the current point to `x`, `y`,
    /// using a control point of `cx`, `cy`
    pub fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.path
            .ops
            .push(PathOp::QuadTo(Point::new(cx, cy), Point::new(x, y)))
    }


    pub fn arc_wedge(&mut self, c: Point, radius: f32, a: Vector, b: Vector) {
        self.move_to(c.x + a.x * radius, c.y + a.y * radius);
        arc(self, c.x, c.y, radius, a, b);
        self.line_to(c.x, c.y);
        self.close();
    }

    /// Adds a cubic bezier from the current point to `x`, `y`,
    /// using control points `cx1`, `cy1` and `cx2`, `cy2`
    pub fn cubic_to(&mut self, cx1: f32, cy1: f32, cx2: f32, cy2: f32, x: f32, y: f32) {
        self.path.ops.push(PathOp::CubicTo(
            Point::new(cx1, cy1),
            Point::new(cx2, cy2),
            Point::new(x, y),
        ))
    }

    /// Closes the current subpath
    pub fn close(&mut self) {
        self.path.ops.push(PathOp::Close)
    }

    /// Completes the current path
    pub fn finish(self) -> (Path, Vec<Vertex>) {
        (self.path, self.vertices)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LineCap {
    Round,
    Square,
    Butt,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LineJoin {
    Round,
    Miter,
    Bevel,
}

fn compute_normal(p0: Point, p1: Point) -> Option<Vector> {
    let ux = p1.x - p0.x;
    let uy = p1.y - p0.y;

    // this could overflow f32. Skia checks for this and
    // uses a double in that situation
    let ulen = ux.hypot(uy);
    if ulen == 0. {
        return None;
    }
    // the normal is perpendicular to the *unit* vector
    Some(Vector::new(-uy / ulen, ux / ulen))
}

fn flip(v: Vector) -> Vector {
    Vector::new(-v.x, -v.y)
}

/* Compute a spline approximation of the arc
centered at xc, yc from the angle a to the angle b

The angle between a and b should not be more than a
quarter circle (pi/2)

The approximation is similar to an approximation given in:
"Approximation of a cubic bezier curve by circular arcs and vice versa"
by Alekas Riškus. However that approximation becomes unstable when the
angle of the arc approaches 0.

This approximation is inspired by a discusion with Boris Zbarsky
and essentially just computes:

  h = 4.0/3.0 * tan ((angle_B - angle_A) / 4.0);

without converting to polar coordinates.

A different way to do this is covered in "Approximation of a cubic bezier
curve by circular arcs and vice versa" by Alekas Riškus. However, the method
presented there doesn't handle arcs with angles close to 0 because it
divides by the perp dot product of the two angle vectors.
*/
fn arc_segment(path: &mut PathBuilder, xc: f32, yc: f32, radius: f32, a: Vector, b: Vector) {
    let r_sin_a = radius * a.y;
    let r_cos_a = radius * a.x;
    let r_sin_b = radius * b.y;
    let r_cos_b = radius * b.x;

    /* bisect the angle between 'a' and 'b' with 'mid' */
    let mut mid = a + b;
    mid /= mid.length();

    /* bisect the angle between 'a' and 'mid' with 'mid2' this is parallel to a
     * line with angle (B - A)/4 */
    let mid2 = a + mid;

    let h = (4. / 3.) * dot(perp(a), mid2) / dot(a, mid2);

    path.cubic_to(
            xc + r_cos_a - h * r_sin_a,
        yc + r_sin_a + h * r_cos_a,
        xc + r_cos_b + h * r_sin_b,
        yc + r_sin_b - h * r_cos_b,
        xc + r_cos_b,
        yc + r_sin_b,
    );
}

fn arc_segment_tri(path: &mut PathBuilder, xc: f32, yc: f32, radius: f32, a: Vector, b: Vector) {
    let r_sin_a = radius * a.y;
    let r_cos_a = radius * a.x;
    let r_sin_b = radius * b.y;
    let r_cos_b = radius * b.x;


    /* bisect the angle between 'a' and 'b' with 'mid' */
    let mut mid = a + b;
    mid /= mid.length();

    /* bisect the angle between 'a' and 'mid' with 'mid2' this is parallel to a
     * line with angle (B - A)/4 */
    let mid2 = a + mid;

    let h = (4. / 3.) * dot(perp(a), mid2) / dot(a, mid2);

    let mut last_point = GpPointR { x: (xc + r_cos_a) as f64, y: (yc + r_sin_a) as f64 };
    

    struct Target<'a> { last_point: GpPointR, xc: f32, yc: f32, path: &'a mut PathBuilder }
    impl<'a> CFlatteningSink for Target<'a> {
        fn AcceptPointAndTangent(&mut self,
            pt: &GpPointR,
                // The point
            vec: &GpPointR,
                // The tangent there
            fLast: bool
                // Is this the last point on the curve?
            ) -> HRESULT {
            todo!()
        }

        fn AcceptPoint(&mut self,
            pt: &GpPointR,
                // The point
            t: f64,
                // Parameter we're at
            fAborted: &mut bool) -> HRESULT {
            self.path.push_tri(self.last_point.x as f32, self.last_point.y as f32, pt.x as f32, pt.y as f32, self.xc, self.yc);
            self.last_point = pt.clone();
            return S_OK;
        }
    }
    let bezier = CBezier::new([GpPointR { x: (xc + r_cos_a) as f64, y: (yc + r_sin_a) as f64,  },
        GpPointR { x: (xc + r_cos_a - h * r_sin_a) as f64, y: (yc + r_sin_a + h * r_cos_a) as f64, },
        GpPointR { x: (xc + r_cos_b + h * r_sin_b) as f64, y: (yc + r_sin_b - h * r_cos_b) as f64, },
        GpPointR { x: (xc + r_cos_b) as f64, y: (yc + r_sin_b) as f64, }]);
        let mut t = Target{ last_point, xc, yc, path };
    let mut f = CBezierFlattener::new(&bezier, &mut t, 0.1);
    f.Flatten(false);

}

/* The angle between the vectors must be <= pi */
fn bisect(a: Vector, b: Vector) -> Vector {
    let mut mid;
    if dot(a, b) >= 0. {
        /* if the angle between a and b is accute, then we can
         * just add the vectors and normalize */
        mid = a + b;
    } else {
        /* otherwise, we can flip a, add it
         * and then use the perpendicular of the result */
        mid = flip(a) + b;
        mid = perp(mid);
    }

    /* normalize */
    /* because we assume that 'a' and 'b' are normalized, we can use
     * sqrt instead of hypot because the range of mid is limited */
    let mid_len = mid.x * mid.x + mid.y * mid.y;
    let len = mid_len.sqrt();
    return mid / len;
}

fn arc(path: &mut PathBuilder, xc: f32, yc: f32, radius: f32, a: Vector, b: Vector) {
    /* find a vector that bisects the angle between a and b */
    let mid_v = bisect(a, b);

    /* construct the arc using two curve segments */
    arc_segment(path, xc, yc, radius, a, mid_v);
    arc_segment(path, xc, yc, radius, mid_v, b);

    arc_segment_tri(path, xc, yc, radius, a, mid_v);
    arc_segment_tri(path, xc, yc, radius, mid_v, b);
}

fn join_round(path: &mut PathBuilder, center: Point, a: Vector, b: Vector, radius: f32) {
    /*
    int ccw = dot (perp (b), a) >= 0; // XXX: is this always true?
    yes, otherwise we have an interior angle.
    assert (ccw);
    */
    arc(path, center.x, center.y, radius, a, b);
}

fn cap_line(dest: &mut PathBuilder, style: &StrokeStyle, pt: Point, normal: Vector) {
    let offset = style.width / 2.;
    match style.cap {
        LineCap::Butt => { /* nothing to do */ }
        LineCap::Round => {
            dest.arc_wedge(pt, offset, normal, flip(normal));
        }
        LineCap::Square => {
            // parallel vector
            let v = Vector::new(normal.y, -normal.x);
            let end = pt + v * offset;
            dest.quad(pt.x + normal.x * offset, pt.y + normal.y * offset,
            end.x + normal.x * offset, end.y + normal.y * offset,
            end.x + -normal.x * offset, end.y + -normal.y * offset,
            pt.x - normal.x * offset, pt.y - normal.y * offset);
        }
    }
}

fn bevel(
    dest: &mut PathBuilder,
    style: &StrokeStyle,
    pt: Point,
    s1_normal: Vector,
    s2_normal: Vector,
) {
    let offset = style.width / 2.;
    dest.tri(pt.x + s1_normal.x * offset, pt.y + s1_normal.y * offset,
          pt.x + s2_normal.x * offset, pt.y + s2_normal.y * offset,
          pt.x, pt.y);
}

/* given a normal rotate the vector 90 degrees to the right clockwise
 * This function has a period of 4. e.g. swap(swap(swap(swap(x) == x */
fn swap(a: Vector) -> Vector {
    /* one of these needs to be negative. We choose a.x so that we rotate to the right instead of negating */
    Vector::new(a.y, -a.x)
}

fn unperp(a: Vector) -> Vector {
    swap(a)
}

/* rotate a vector 90 degrees to the left */
fn perp(v: Vector) -> Vector {
    Vector::new(-v.y, v.x)
}

fn dot(a: Vector, b: Vector) -> f32 {
    a.x * b.x + a.y * b.y
}

/* Finds the intersection of two lines each defined by a point and a normal.
From "Example 2: Find the intersection of two lines" of
"The Pleasures of "Perp Dot" Products"
F. S. Hill, Jr. */
fn line_intersection(a: Point, a_perp: Vector, b: Point, b_perp: Vector) -> Option<Point> {
    let a_parallel = unperp(a_perp);
    let c = b - a;
    let denom = dot(b_perp, a_parallel);
    if denom == 0.0 {
        return None;
    }

    let t = dot(b_perp, c) / denom;

    let intersection = Point::new(a.x + t * (a_parallel.x), a.y + t * (a_parallel.y));

    Some(intersection)
}

fn is_interior_angle(a: Vector, b: Vector) -> bool {
    /* angles of 180 and 0 degress will evaluate to 0, however
     * we to treat 180 as an interior angle and 180 as an exterior angle */
    dot(perp(a), b) > 0. || a == b /* 0 degrees is interior */
}

fn join_line(
    dest: &mut PathBuilder,
    style: &StrokeStyle,
    pt: Point,
    mut s1_normal: Vector,
    mut s2_normal: Vector,
) {
    if is_interior_angle(s1_normal, s2_normal) {
        s2_normal = flip(s2_normal);
        s1_normal = flip(s1_normal);
        std::mem::swap(&mut s1_normal, &mut s2_normal);
    }

    // XXX: joining uses `pt` which can cause seams because it lies halfway on a line and the
    // rasterizer may not find exactly the same spot
    let offset = style.width / 2.;
    match style.join {
        LineJoin::Round => {
            dest.arc_wedge(pt, offset, s1_normal, s2_normal);
        }
        LineJoin::Miter => {
            let in_dot_out = -s1_normal.x * s2_normal.x + -s1_normal.y * s2_normal.y;
            if 2. <= style.miter_limit * style.miter_limit * (1. - in_dot_out) {
                let start = pt + s1_normal * offset;
                let end = pt + s2_normal * offset;
                if let Some(intersection) = line_intersection(start, s1_normal, end, s2_normal) {
                    // We won't have an intersection if the segments are parallel
                    dest.quad(pt.x + s1_normal.x * offset, pt.y + s1_normal.y * offset,
                    intersection.x, intersection.y,
                    pt.x + s2_normal.x * offset, pt.y + s2_normal.y * offset,
                    pt.x, pt.y);
                }
            } else {
                bevel(dest, style, pt, s1_normal, s2_normal);
            }
        }
        LineJoin::Bevel => {
            bevel(dest, style, pt, s1_normal, s2_normal);
        }
    }
}

pub fn stroke_to_path(path: &Path, style: &StrokeStyle) -> (Path, Vec<Vertex>) {
    let mut stroked_path = PathBuilder::new();

    if style.width <= 0. {
        return stroked_path.finish();
    }

    let mut cur_pt = None;
    let mut last_normal = Vector::zero();
    let half_width = style.width / 2.;
    let mut start_point = None;
    for op in &path.ops {
        match *op {
            PathOp::MoveTo(pt) => {
                if let (Some(cur_pt), Some((point, normal))) = (cur_pt, start_point) {
                    // cap end
                    cap_line(&mut stroked_path, style, cur_pt, last_normal);
                    // cap beginning
                    cap_line(&mut stroked_path, style, point, flip(normal));
                }
                start_point = None;
                cur_pt = Some(pt);
            }
            PathOp::LineTo(pt) => {
                if cur_pt.is_none() {
                    start_point = None;
                } else if let Some(cur_pt) = cur_pt {
                    if let Some(normal) = compute_normal(cur_pt, pt) {
                        if start_point.is_none() {
                            start_point = Some((cur_pt, normal));
                        } else {
                            join_line(&mut stroked_path, style, cur_pt, last_normal, normal);
                        }
                        if (true) {
                            stroked_path.ramp(                        

                                pt.x + normal.x * (half_width - 0.5), 
                                pt.y + (normal.y * half_width - 0.5),
                                pt.x + normal.x * (half_width + 0.5),
                                pt.y + normal.y * (half_width + 0.5),
                                cur_pt.x + normal.x * (half_width + 0.5),
                                cur_pt.y + normal.y * (half_width + 0.5),
                                cur_pt.x + normal.x * (half_width - 0.5),
                                cur_pt.y + normal.y * (half_width - 0.5),
                            );
                            stroked_path.quad(
                                cur_pt.x + normal.x * (half_width - 0.5),
                                cur_pt.y + normal.y * (half_width - 0.5),
                                pt.x + normal.x * (half_width - 0.5), pt.y + normal.y * (half_width - 0.5),
                                pt.x + -normal.x * (half_width - 0.5), pt.y + -normal.y * (half_width - 0.5),
                                cur_pt.x - normal.x * (half_width - 0.5),
                                cur_pt.y - normal.y * (half_width - 0.5),
                            );
                            stroked_path.ramp(                        
                                cur_pt.x - normal.x * (half_width - 0.5),
                                cur_pt.y - normal.y * (half_width - 0.5),
                                cur_pt.x - normal.x * (half_width + 0.5),
                                cur_pt.y - normal.y * (half_width + 0.5),
                                pt.x - normal.x * (half_width + 0.5),
                                pt.y - normal.y * (half_width + 0.5),
                                pt.x - normal.x * (half_width - 0.5), 
                                pt.y - (normal.y * half_width - 0.5),
                            );
                        } else {
                            stroked_path.quad(
                                cur_pt.x + normal.x * half_width,
                                cur_pt.y + normal.y * half_width,
                                pt.x + normal.x * half_width, pt.y + normal.y * half_width,
                                pt.x + -normal.x * half_width, pt.y + -normal.y * half_width,
                                cur_pt.x - normal.x * half_width,
                                cur_pt.y - normal.y * half_width,
                            );
                        }

                        last_normal = normal;

                    }
                }
                cur_pt = Some(pt);

            }
            PathOp::Close => {
                if let (Some(cur_pt), Some((end_point, start_normal))) = (cur_pt, start_point) {
                    if let Some(normal) = compute_normal(cur_pt, end_point) {
                        join_line(&mut stroked_path, style, cur_pt, last_normal, normal);

                        stroked_path.quad(
                            cur_pt.x + normal.x * half_width,
                            cur_pt.y + normal.y * half_width,
                            end_point.x + normal.x * half_width,
                            end_point.y + normal.y * half_width,
                            end_point.x + -normal.x * half_width,
                            end_point.y + -normal.y * half_width,
                            cur_pt.x - normal.x * half_width,
                            cur_pt.y - normal.y * half_width,
                        );
                        join_line(&mut stroked_path, style, end_point, normal, start_normal);
                    } else {
                        join_line(&mut stroked_path, style, end_point, last_normal, start_normal);
                    }
                }
                cur_pt = start_point.map(|x| x.0);
                start_point = None;
            }
            PathOp::QuadTo(..) => panic!("Only flat paths handled"),
            PathOp::CubicTo(..) => panic!("Only flat paths handled"),
        }
    }
    if let (Some(cur_pt), Some((point, normal))) = (cur_pt, start_point) {
        // cap end
        cap_line(&mut stroked_path, style, cur_pt, last_normal);
        // cap beginning
        cap_line(&mut stroked_path, style, point, flip(normal));
    }
    stroked_path.finish()
}


fn write_image(data: &[u8], path: &str, width: u32, height: u32) {
    use std::path::Path;
    use std::fs::File;
    use std::io::BufWriter;

    /*let mut png_data: Vec<u8> = vec![0; (width * height * 3) as usize];
    let mut i = 0;
    for pixel in data {
        png_data[i] = ((pixel >> 16) & 0xff) as u8;
        png_data[i + 1] = ((pixel >> 8) & 0xff) as u8;
        png_data[i + 2] = ((pixel >> 0) & 0xff) as u8;
        i += 3;
    }*/


    let path = Path::new(path);
    let file = File::create(path).unwrap();
    let w = &mut BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, width, height); // Width is 2 pixels and height is 1.
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();

    writer.write_image_data(&data).unwrap(); // Save
}


// How do we handle transformed paths?
fn main() {
    let mut p = PathBuilder::new();
    p.move_to(10., 10.);
    p.line_to(100., 100.);
    p.line_to(110., 10.);

    let path = p.finish().0;
    let stroked = stroke_to_path(&path, &StrokeStyle{
        cap: LineCap::Round, 
        join: LineJoin::Round, 
        width: 10.,
         ..Default::default()});
    dbg!(&stroked);

    let mask = rasterize_to_mask(&stroked.1, 200, 200);
    write_image(&mask,"out.png", 200, 200);
/* 
    struct Target;
    impl CFlatteningSink for Target {
        fn AcceptPointAndTangent(&mut self,
        pt: &GpPointR,
            // The point
        vec: &GpPointR,
            // The tangent there
        fLast: bool
            // Is this the last point on the curve?
        ) -> HRESULT {
        todo!()
    }

        fn AcceptPoint(&mut self,
            pt: &GpPointR,
                // The point
            t: f64,
                // Parameter we're at
            fAborted: &mut bool) -> HRESULT {
        println!("{} {}", pt.x, pt.y);
        return S_OK;
    }
    }
    let bezier = CBezier::new([GpPointR { x: 0., y: 0. },
        GpPointR { x: 0., y: 0. },
        GpPointR { x: 0., y: 0. },
        GpPointR { x: 100., y: 100. }]);
        let mut t = Target{};
    let mut f = CBezierFlattener::new(&bezier, &mut t, 0.1);
    f.Flatten(false);*/

}
