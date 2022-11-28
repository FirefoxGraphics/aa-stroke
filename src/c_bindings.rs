use crate::{Stroker, StrokeStyle, Point};

type OutputVertex = crate::Vertex;

#[repr(C)]
pub struct VertexBuffer {
    data: *const OutputVertex,
    len: usize
}

#[no_mangle]
pub extern "C" fn aa_stroke_new(style: &StrokeStyle) -> *mut Stroker {
    let s = Stroker::new(style);
    Box::into_raw(Box::new(s))
}

#[no_mangle]
pub extern "C" fn aa_stroke_start_sub_path(s: &mut Stroker, x: f32, y: f32, closed: bool) {
    s.start_sub_path(Point::new(x, y), closed);
}

#[no_mangle]
pub extern "C" fn aa_stroke_cap_sub_path(s: &mut Stroker, x: f32, y: f32) {
    s.cap_sub_path(Point::new(x, y));
}

#[no_mangle]
pub extern "C" fn aa_stroke_line_to(s: &mut Stroker, x: f32, y: f32) {
    s.line_to(Point::new(x, y));
}

/*#[no_mangle]
pub extern "C" fn aa_stroke_curve_to(s: &mut Stroker, c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) {
    s.curve_to(c1x, c1y, c2x, c2y, x, y);
}

#[no_mangle]
pub extern "C" fn aa_stroke_quad_to(s: &mut Stroker, cx: f32, cy: f32, x: f32, y: f32) {
    s.quad_to(cx, cy, x, y);
}*/

#[no_mangle]
pub extern "C" fn aa_stroke_close(s: &mut Stroker) {
    s.close();
}

#[no_mangle]
pub extern "C" fn aa_stroke_finish(s: &mut Stroker) -> VertexBuffer {
    let result = s.finish().1;
    let vb = VertexBuffer { data: result.as_ptr(), len: result.len() };
    std::mem::forget(result);
    vb
}

#[no_mangle]
pub extern "C" fn aa_stroke_vertex_buffer_release(vb: VertexBuffer)
{
    unsafe { drop(Box::from_raw(std::slice::from_raw_parts_mut(vb.data as *mut OutputVertex, vb.len))) }
}

#[no_mangle]
pub unsafe extern "C" fn aa_stroke_release(s: *mut Stroker) {
    drop(Box::from_raw(s));
}


#[test]
fn simple() {
    let style = StrokeStyle::default();
    let s = unsafe { &mut *aa_stroke_new(&style) } ;
    aa_stroke_start_sub_path(s, 10., 10., false);
    aa_stroke_line_to(s, 100., 100.);
    aa_stroke_cap_sub_path(s, 100., 10.);
    let vb = aa_stroke_finish(s);
    aa_stroke_vertex_buffer_release(vb);
    unsafe { aa_stroke_release(s) } ;
}