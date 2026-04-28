# ESP32 Hacking

## FPS over COM

Use the serial FPS log as the measurement source, not the LCD overlay. The LCD
is useful for a quick sanity check, but the COM log includes the average invoke,
push, and total frame times.

Known-good command from `devices/Waveshare_ESP32_C6`:

```powershell
.\scripts\monitor.cmd COM4 -Seconds 8 -FpsOnly
```

For full boot and runtime output:

```powershell
.\scripts\monitor.cmd COM4 -Seconds 15
```

Example line:

```text
wasm mandelbrot frame 666: 22 fps invoke=40321us push=4634us total=44956us n=23
```

## Why plain espflash monitor was misleading

On the Waveshare ESP32-C6 LCD board, `COM4` is the USB Serial/JTAG device
(`303A:1001`). `espflash monitor` can connect and print its command help, but it
may still show no app-level `println!` output. Raw COM reads with the wrong
control-line state can also show nothing.

The firmware must instantiate and keep the HAL USB Serial/JTAG peripheral alive:

```rust
let serial_console = SerialConsole {
    _usb: UsbSerialJtag::new(peripherals.USB_DEVICE),
};
```

Without that, the bootloader can still print, while the app's once-per-second
FPS `println!` lines do not reliably reach COM. Keeping the `UsbSerialJtag`
value alive is enough; the existing `esp_println::println!` calls then produce
the FPS lines.

The local `monitor.ps1` intentionally reads `System.IO.Ports.SerialPort`
directly at 115200 baud with `DtrEnable = false` and `RtsEnable = true`. That is
the control-line combination that exposed the runtime stream during testing.

## If COM output disappears again

1. Flash without the monitor first:

```powershell
.\scripts\flash.cmd COM4 -NoMonitor --release
```

2. Close any existing serial monitor, then run:

```powershell
.\scripts\monitor.cmd COM4 -Seconds 15
```

3. If you only see bootloader lines or no app lines, check that
`src/main.rs` still creates and retains `UsbSerialJtag::new(peripherals.USB_DEVICE)`.

4. Prefer `-FpsOnly` for measurements once full output is known to work:

```powershell
.\scripts\monitor.cmd COM4 -Seconds 8 -FpsOnly
```
