//! Board constants for the Waveshare ESP32-C6-LCD-1.47.

pub const LCD_WIDTH: usize = 320;
pub const LCD_HEIGHT: usize = 172;

pub const LCD_X_OFFSET: u16 = 0;
pub const LCD_Y_OFFSET: u16 = 34;

pub const LCD_MADCTL: u8 = 0x60;

pub const LCD_SPI_MOSI_GPIO: u8 = 6;
pub const LCD_SPI_SCLK_GPIO: u8 = 7;
pub const LCD_CS_GPIO: u8 = 14;
pub const LCD_DC_GPIO: u8 = 15;
pub const LCD_RST_GPIO: u8 = 21;
pub const LCD_BL_GPIO: u8 = 22;

pub const RGB_LED_GPIO: u8 = 8;
pub const TF_CARD_CS_GPIO: u8 = 4;
pub const TF_CARD_MISO_GPIO: u8 = 5;
