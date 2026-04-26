//! Spinning solid cube — "early DirectX demo" style.
//!
//! Six colored faces (R, G, B, Y, C, M), flat-shaded with a fixed
//! light-at-camera, tumbling on two axes. All i32 throughout:
//! - Vertex coords: integer screen-ish units.
//! - Trig: 64-entry Q15 LUT (`cos(2π·i/64) × 32767`).
//! - Rotation: 2D rotations chained (yaw around Y, then pitch around
//!   X). Product `coord × Q15 >> 15` stays comfortably inside i32.
//! - Projection: pinhole, integer divide by camera-space Z.
//! - Raster: edge-function triangle scan with incremental updates.

pub const WIDTH: usize = 160;
pub const HEIGHT: usize = 128;
pub const FB_BYTES: usize = WIDTH * HEIGHT * 2;

// ── Q15 cosine LUT, proper 64 entries ────────────────────────────────
//
// cos(2π·i/64) × 32767, one entry per 5.625°. Earlier revision was a
// 32-entry table duplicated twice that matched `cos_q15` but broke
// `sin_q15`, which expects the π/2 offset to be 16 indices (true only
// for a 64-entry LUT). That mismatch produced a rotation that
// collapsed into non-uniform scaling with periodic sign flips.
static COS_Q15: [i32; 64] = [
     32767,  32610,  32138,  31357,  30274,  28899,  27246,  25330,
     23170,  20788,  18205,  15447,  12540,   9512,   6393,   3212,
         0,  -3212,  -6393,  -9512, -12540, -15447, -18205, -20788,
    -23170, -25330, -27246, -28899, -30274, -31357, -32138, -32610,
    -32767, -32610, -32138, -31357, -30274, -28899, -27246, -25330,
    -23170, -20788, -18205, -15447, -12540,  -9512,  -6393,  -3212,
         0,   3212,   6393,   9512,  12540,  15447,  18205,  20788,
     23170,  25330,  27246,  28899,  30274,  31357,  32138,  32610,
];

/// Cosine with 16 sub-steps per LUT entry. The input unit is
/// 2π / (64 * 16), which lets slow rotations advance every frame
/// without jumping a whole 5.625-degree LUT step at a time.
#[inline(always)]
fn cos_q15(angle_x16: i32) -> i32 {
    let idx = (angle_x16 >> 4) & 63;
    let frac = angle_x16 & 15;
    let a = COS_Q15[idx as usize];
    let b = COS_Q15[((idx + 1) & 63) as usize];
    a + (((b - a) * frac) >> 4)
}

#[inline(always)]
fn sin_q15(angle_x16: i32) -> i32 {
    // sin(a) = cos(a - π/2). In x16 LUT units, π/2 is 16 * 16.
    cos_q15(angle_x16 - 256)
}

// ── Cube geometry ───────────────────────────────────────────────────
type V3 = (i32, i32, i32);

/// Half-size of the cube in world units. Chosen together with
/// `CAMERA_Z` and `FOCAL` so the cube fills ~70% of the 160×128
/// panel without clipping at full tumble — large enough that face
/// colors and rotation are obvious, small enough that the farthest
/// corner at 45° yaw+pitch stays within the 80-px screen half-width.
const HALF: i32 = 36;
const CAMERA_Z: i32 = 230;
const FOCAL: i32 = 200;
const CX: i32 = (WIDTH as i32) / 2;
const CY: i32 = (HEIGHT as i32) / 2;

const VERTS: [V3; 8] = [
    (-HALF, -HALF, -HALF), // 0
    ( HALF, -HALF, -HALF), // 1
    ( HALF,  HALF, -HALF), // 2
    (-HALF,  HALF, -HALF), // 3
    (-HALF, -HALF,  HALF), // 4
    ( HALF, -HALF,  HALF), // 5
    ( HALF,  HALF,  HALF), // 6
    (-HALF,  HALF,  HALF), // 7
];

/// Each face lists its 4 corners in CCW order viewed from outside.
/// We triangulate as `(a, b, c)` + `(a, c, d)` at draw time.
const FACES: [[u8; 4]; 6] = [
    [0, 1, 2, 3], // -Z  (nearest face when un-rotated)
    [5, 4, 7, 6], // +Z  (farthest face when un-rotated)
    [1, 5, 6, 2], // +X
    [4, 0, 3, 7], // -X
    [4, 5, 1, 0], // -Y
    [3, 2, 6, 7], // +Y
];

/// Outward face normals in Q15. We rotate these alongside the
/// vertices and use their Z component for both back-face culling
/// and flat shading (light points along +Z from the camera).
const FACE_NORMALS: [V3; 6] = [
    (0, 0, -32767),
    (0, 0,  32767),
    ( 32767, 0, 0),
    (-32767, 0, 0),
    (0, -32767, 0),
    (0,  32767, 0),
];

/// Classic "demo cube" palette — one color per face axis.
const FACE_COLORS: [u16; 6] = [
    0xF800, // -Z red
    0x07E0, // +Z green
    0x001F, // +X blue
    0xFFE0, // -X yellow
    0x07FF, // -Y cyan
    0xF81F, // +Y magenta
];

// ── Rotation / projection ───────────────────────────────────────────
#[inline(always)]
fn rotate(v: V3, yc: i32, ys: i32, pc: i32, ps: i32) -> V3 {
    let (x, y, z) = v;
    // Yaw around Y: X ↔ Z mixing, Y unchanged.
    let x1 = (x * yc - z * ys) >> 15;
    let z1 = (x * ys + z * yc) >> 15;
    // Pitch around X: Y ↔ (rotated)Z mixing, X unchanged.
    let y2 = (y * pc - z1 * ps) >> 15;
    let z2 = (y * ps + z1 * pc) >> 15;
    (x1, y2, z2)
}

#[inline(always)]
fn project(v: V3) -> (i32, i32) {
    let (x, y, z) = v;
    // Camera at origin looking +Z; cube lives at z ≈ CAMERA_Z.
    let zr = z + CAMERA_Z;
    let zr = if zr < 1 { 1 } else { zr }; // trivial near-clip
    (CX + (x * FOCAL) / zr, CY + (y * FOCAL) / zr)
}

// ── Shading ─────────────────────────────────────────────────────────
/// Scale an RGB565 color by a Q15 brightness, then lift by a small
/// ambient floor so faces that turn edge-on don't collapse to black.
#[inline(always)]
fn shade(base: u16, brightness_q15: i32) -> u16 {
    let r = ((base >> 11) & 0x1F) as i32;
    let g = ((base >> 5) & 0x3F) as i32;
    let b = (base & 0x1F) as i32;
    // 15% ambient + 85% directional — high contrast between the face
    // most-facing the camera and faces turning edge-on. That changing
    // brightness is itself a strong 3D-rotation cue.
    let mut lit = 4915 + ((brightness_q15 * 27852) >> 15);
    if lit > 32767 { lit = 32767; }
    if lit < 0 { lit = 0; }
    let r = (r * lit >> 15) as u16;
    let g = (g * lit >> 15) as u16;
    let b = (b * lit >> 15) as u16;
    (r << 11) | (g << 5) | b
}

// ── Triangle rasterizer ─────────────────────────────────────────────
/// Flat-fill a triangle using edge-function coverage. Winding is
/// auto-corrected so either CCW or CW input draws the same triangle.
fn fill_triangle(
    bytes: &mut [u8],
    v0: (i32, i32),
    v1: (i32, i32),
    v2: (i32, i32),
    color: u16,
) {
    let (x0, y0) = v0;
    let (mut x1, mut y1) = v1;
    let (mut x2, mut y2) = v2;

    // Signed 2× area. Swap v1/v2 if negative so we can always use the
    // "e ≥ 0 means inside" convention for the edge-function test.
    let area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if area2 == 0 {
        return;
    }
    if area2 < 0 {
        core::mem::swap(&mut x1, &mut x2);
        core::mem::swap(&mut y1, &mut y2);
    }

    // Bounding box clipped to the panel.
    let min_x = x0.min(x1).min(x2).max(0);
    let max_x = x0.max(x1).max(x2).min(WIDTH as i32 - 1);
    let min_y = y0.min(y1).min(y2).max(0);
    let max_y = y0.max(y1).max(y2).min(HEIGHT as i32 - 1);
    if min_x > max_x || min_y > max_y {
        return;
    }

    // Edge functions:
    //   e(v_a, v_b)(x, y) = (x_b − x_a)(y − y_a) − (y_b − y_a)(x − x_a)
    // Step per +x: −(y_b − y_a) = y_a − y_b
    // Step per +y:  (x_b − x_a)
    let dex0 = y0 - y1; let dey0 = x1 - x0;
    let dex1 = y1 - y2; let dey1 = x2 - x1;
    let dex2 = y2 - y0; let dey2 = x0 - x2;

    // Initial edge values at (min_x, min_y).
    let mut row_e0 = (x1 - x0) * (min_y - y0) - (y1 - y0) * (min_x - x0);
    let mut row_e1 = (x2 - x1) * (min_y - y1) - (y2 - y1) * (min_x - x1);
    let mut row_e2 = (x0 - x2) * (min_y - y2) - (y0 - y2) * (min_x - x2);

    let hi = (color >> 8) as u8;
    let lo = color as u8;

    let mut y = min_y;
    while y <= max_y {
        let mut e0 = row_e0;
        let mut e1 = row_e1;
        let mut e2 = row_e2;
        let mut x = min_x;
        while x <= max_x {
            // `e0 | e1 | e2 >= 0` iff none of the three have their
            // sign bit set, i.e. all are ≥ 0 — one branch per pixel.
            if (e0 | e1 | e2) >= 0 {
                let idx = (y as usize * WIDTH + x as usize) * 2;
                bytes[idx] = hi;
                bytes[idx + 1] = lo;
            }
            e0 += dex0;
            e1 += dex1;
            e2 += dex2;
            x += 1;
        }
        row_e0 += dey0;
        row_e1 += dey1;
        row_e2 += dey2;
        y += 1;
    }
}

// ── Per-frame render ────────────────────────────────────────────────
pub fn render(bytes: &mut [u8], frame: u32) {
    debug_assert_eq!(bytes.len(), WIDTH * HEIGHT * 2);

    // Clear to black. `.fill(0)` compiles to a tight memset on both
    // native and JIT.
    bytes.fill(0);

    let frame = frame as i32;
    // Angles are x16 LUT units. This preserves the original slow tumble
    // speed, but removes the old 4-frame/8-frame hold before each
    // whole-LUT-index jump.
    let yaw_x16 = frame << 1;
    let pitch_x16 = frame;
    let yc = cos_q15(yaw_x16);
    let ys = sin_q15(yaw_x16);
    let pc = cos_q15(pitch_x16);
    let ps = sin_q15(pitch_x16);

    // Rotate + project all 8 vertices once; each face reuses the
    // projected points by index.
    let mut proj = [(0i32, 0i32); 8];
    let mut i = 0;
    while i < 8 {
        let rotated = rotate(VERTS[i], yc, ys, pc, ps);
        proj[i] = project(rotated);
        i += 1;
    }

    // Draw each non-back-facing face as two triangles.
    let mut face_i = 0;
    while face_i < 6 {
        let n_rot = rotate(FACE_NORMALS[face_i], yc, ys, pc, ps);
        // Camera looks +Z from origin, so a face is visible when its
        // outward normal points back toward us (z component < 0).
        if n_rot.2 < 0 {
            // Brightness = how directly the face aims at the camera.
            // `-n_rot.z` is already positive Q15 in that range.
            let color = shade(FACE_COLORS[face_i], -n_rot.2);
            let f = FACES[face_i];
            let v0 = proj[f[0] as usize];
            let v1 = proj[f[1] as usize];
            let v2 = proj[f[2] as usize];
            let v3 = proj[f[3] as usize];
            fill_triangle(bytes, v0, v1, v2, color);
            fill_triangle(bytes, v0, v2, v3, color);
        }
        face_i += 1;
    }
}
