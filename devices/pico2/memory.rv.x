/* Memory map for Raspberry Pi Pico 2 / Pico 2 W (RP2350), Hazard3 RV mode.
 *
 * Same physical layout as the ARM build (`memory.arm.x`); the differences
 * are linker-script naming conventions and where `.start_block` sits.
 *
 * - cortex-m-rt's link.x consumes FLASH/RAM directly and ships a
 *   `.vector_table` section at the start of flash to anchor `INSERT
 *   AFTER`.
 * - riscv-rt's link.x consumes REGION_TEXT/REGION_RODATA/REGION_DATA/
 *   REGION_BSS/REGION_HEAP/REGION_STACK and has no `.vector_table` —
 *   `.text` itself is the first thing in REGION_TEXT.
 *
 * For the RP2350 boot ROM to recognize this image, the IMAGE_DEF block
 * (in `.start_block`) must live in the first 4 KiB of flash. Under the
 * RV layout we reserve 256 bytes at the start of flash for it and move
 * `_stext` past those bytes; the `.text` section then lands right after
 * the boot block.
 *
 * Naming-convention shim: rp_binary_info references `__sidata` /
 * `__sdata` / `__edata` (cortex-m-rt-style double underscore). riscv-rt
 * provides single-underscore equivalents. Alias one to the other so the
 * link succeeds without patching rp_binary_info.
 */

MEMORY {
    /* 4 MiB QSPI flash on Pico 2 / Pico 2 W, mapped into XIP at 0x10000000. */
    FLASH : ORIGIN = 0x10000000, LENGTH = 4096K

    /* SRAM banks 0-7, 512 KiB total, striped across banks for bandwidth. */
    RAM   : ORIGIN = 0x20000000, LENGTH = 512K

    /* SRAM banks 8 and 9 use direct (non-striped) mapping. Reserved for
     * future per-core-dedicated regions; unused in milestone 1. */
    SRAM8 : ORIGIN = 0x20080000, LENGTH = 4K
    SRAM9 : ORIGIN = 0x20081000, LENGTH = 4K
}

/* riscv-rt expects REGION_* names, not FLASH/RAM. Alias all of them onto
 * the physical regions defined above. */
REGION_ALIAS("REGION_TEXT",   FLASH);
REGION_ALIAS("REGION_RODATA", FLASH);
REGION_ALIAS("REGION_DATA",   RAM);
REGION_ALIAS("REGION_BSS",    RAM);
REGION_ALIAS("REGION_HEAP",   RAM);
REGION_ALIAS("REGION_STACK",  RAM);

/* Per-hart native stack. The 320 KiB Rust heap (in .bss via
 * embedded-alloc) + 128 KiB JIT code arena + other .data/.bss already
 * consume ~460 KiB of the 512 KiB RAM, leaving roughly 52 KiB for stack
 * + heap-region (we keep _heap_size = 0 since embedded-alloc owns the
 * real heap). 32 KiB is a comfortable native stack for sf-nano-core's
 * compile path; the Wasm operand stack (config::WASM_STACK_BYTES, also
 * 32 KiB) is allocated separately from the embedded-alloc heap.
 *
 * The ARM build sidesteps this by using `flip-link`, which puts the
 * stack at the top of RAM growing down so an overflow faults instead
 * of corrupting .bss. flip-link only supports cortex-m-rt today, so
 * the RV build relies on the explicit reservation below. */
PROVIDE(_hart_stack_size = 32K);

/* Reserve 256 bytes at the start of flash for the boot block. The
 * IMAGE_DEF struct in rp235x-hal is well under 64 bytes; this leaves
 * comfortable headroom and stays well within the boot ROM's 4 KiB cap.
 * `_stext` overrides riscv-rt's default of `ORIGIN(REGION_TEXT)` so
 * `.text` lands after the boot block instead of at the very same
 * address. */
_stext = ORIGIN(REGION_TEXT) + 0x100;

/* Place `.start_block` at the very start of flash, before riscv-rt's
 * `.text.dummy` advances the location counter to `_stext`. The
 * `.text.dummy (NOLOAD)` section then accounts for the 256-byte gap
 * without emitting any bytes itself. */
SECTIONS {
    .start_block ORIGIN(REGION_TEXT) : ALIGN(4)
    {
        __start_block_addr = .;
        KEEP(*(.start_block));
        KEEP(*(.boot_info));
    } > FLASH
} INSERT BEFORE .text.dummy;

/* Picotool binary-info records, pointed at by IMAGE_DEF. */
SECTIONS {
    .bi_entries : ALIGN(4)
    {
        __bi_entries_start = .;
        KEEP(*(.bi_entries));
        . = ALIGN(4);
        __bi_entries_end = .;
    } > FLASH
} INSERT AFTER .text;

/* Trailing boot block; used by picotool for image signatures. */
SECTIONS {
    .end_block : ALIGN(4)
    {
        __end_block_addr = .;
        KEEP(*(.end_block));
        __flash_binary_end = .;
    } > FLASH
} INSERT AFTER .bi_entries;

PROVIDE(start_to_end = __end_block_addr - __start_block_addr);
PROVIDE(end_to_start = __start_block_addr - __end_block_addr);

/* rp_binary_info uses cortex-m-rt-style symbol names for the data-init
 * region; alias them onto riscv-rt's single-underscore equivalents. */
PROVIDE(__sidata = _sidata);
PROVIDE(__sdata  = _sdata);
PROVIDE(__edata  = _edata);
