/* Memory map for Raspberry Pi Pico 2 / Pico 2 W (RP2350). */

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

SECTIONS {
    /* Boot ROM header: must live in the first 4 KiB of flash, after
     * .vector_table, so the RP2350 Boot ROM and picotool can find the
     * IMAGE_DEF block. rp235x-hal's `hal::block::ImageDef` goes here. */
    .start_block : ALIGN(4)
    {
        __start_block_addr = .;
        KEEP(*(.start_block));
        KEEP(*(.boot_info));
    } > FLASH

} INSERT AFTER .vector_table;

/* .text starts right after the boot header. */
_stext = ADDR(.start_block) + SIZEOF(.start_block);

SECTIONS {
    /* Picotool binary-info records. Pointed at by the IMAGE_DEF so
     * `picotool info` can introspect build metadata. */
    .bi_entries : ALIGN(4)
    {
        __bi_entries_start = .;
        KEEP(*(.bi_entries));
        . = ALIGN(4);
        __bi_entries_end = .;
    } > FLASH
} INSERT AFTER .text;

SECTIONS {
    /* Trailing boot block; used by picotool for image signatures. */
    .end_block : ALIGN(4)
    {
        __end_block_addr = .;
        KEEP(*(.end_block));
        __flash_binary_end = .;
    } > FLASH

} INSERT AFTER .uninit;

PROVIDE(start_to_end = __end_block_addr - __start_block_addr);
PROVIDE(end_to_start = __start_block_addr - __end_block_addr);
