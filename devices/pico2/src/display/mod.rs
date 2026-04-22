//! Display subsystem — ST7735s init, framebuffer wrapper, and the
//! FPS overlay.
//!
//! Each submodule is usable independently; the three are grouped here
//! because the binaries that touch the panel typically use all of them
//! together.

pub mod framebuffer;
pub mod overlay;
pub mod st7735;

pub use framebuffer::Framebuffer;
pub use overlay::{format_fps, stamp_fps_overlay};
pub use st7735::{
    init, write_cmd, write_data, CASET_DATA, CMD_CASET, CMD_RAMWR, CMD_RASET, RASET_DATA,
};
