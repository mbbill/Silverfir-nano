/* Standalone linker script for Raspberry Pi Pico 2 / Pico 2 W (RP2350),
 * Hazard3 RV32IMAC mode.
 *
 * This follows the upstream rp235x-hal RISC-V example layout: the riscv-rt
 * reset code starts at 0x1000_0000, and the RP2350 IMAGE_DEF block is kept
 * inside .text shortly after that reset stub. The boot ROM scans the first
 * 4 KiB for IMAGE_DEF, but when no explicit entry-point item is present it
 * expects execution to begin at the image start. Keeping _start at flash
 * origin therefore matters.
 *
 * This file replaces riscv-rt's link.x for RV builds. It is based on
 * riscv-rt 0.12's link.x plus the RP2350 boot metadata sections used by
 * rp235x-hal-examples.
 */

MEMORY {
    /* 4 MiB QSPI flash on Pico 2 / Pico 2 W, mapped into XIP at 0x10000000. */
    FLASH : ORIGIN = 0x10000000, LENGTH = 4096K

    /* SRAM banks 0-7, 512 KiB total, striped across banks for bandwidth. */
    RAM   : ORIGIN = 0x20000000, LENGTH = 512K

    /* SRAM banks 8 and 9 use direct mapping. */
    SRAM8 : ORIGIN = 0x20080000, LENGTH = 4K
    SRAM9 : ORIGIN = 0x20081000, LENGTH = 4K
}

PROVIDE(_stext = ORIGIN(FLASH));
PROVIDE(_stack_start = ORIGIN(RAM) + LENGTH(RAM));
PROVIDE(_max_hart_id = 0);

/* The native stack must coexist with the static Rust heap and the JIT code
 * arena. The current RV demo leaves a little over 30 KiB after static
 * storage, code arena, and Wasm memory reservations. */
PROVIDE(_hart_stack_size = 30K);
PROVIDE(_heap_size = 0);

PROVIDE(InstructionMisaligned = ExceptionHandler);
PROVIDE(InstructionFault = ExceptionHandler);
PROVIDE(IllegalInstruction = ExceptionHandler);
PROVIDE(Breakpoint = ExceptionHandler);
PROVIDE(LoadMisaligned = ExceptionHandler);
PROVIDE(LoadFault = ExceptionHandler);
PROVIDE(StoreMisaligned = ExceptionHandler);
PROVIDE(StoreFault = ExceptionHandler);
PROVIDE(UserEnvCall = ExceptionHandler);
PROVIDE(SupervisorEnvCall = ExceptionHandler);
PROVIDE(MachineEnvCall = ExceptionHandler);
PROVIDE(InstructionPageFault = ExceptionHandler);
PROVIDE(LoadPageFault = ExceptionHandler);
PROVIDE(StorePageFault = ExceptionHandler);

PROVIDE(SupervisorSoft = DefaultHandler);
PROVIDE(MachineSoft = DefaultHandler);
PROVIDE(SupervisorTimer = DefaultHandler);
PROVIDE(MachineTimer = DefaultHandler);
PROVIDE(SupervisorExternal = DefaultHandler);
PROVIDE(MachineExternal = DefaultHandler);

PROVIDE(DefaultHandler = DefaultInterruptHandler);
PROVIDE(ExceptionHandler = DefaultExceptionHandler);

PROVIDE(__pre_init = default_pre_init);
PROVIDE(_setup_interrupts = default_setup_interrupts);

PROVIDE(TIMER0_IRQ_0 = DefaultHandler);
PROVIDE(TIMER0_IRQ_1 = DefaultHandler);
PROVIDE(TIMER0_IRQ_2 = DefaultHandler);
PROVIDE(TIMER0_IRQ_3 = DefaultHandler);
PROVIDE(TIMER1_IRQ_0 = DefaultHandler);
PROVIDE(TIMER1_IRQ_1 = DefaultHandler);
PROVIDE(TIMER1_IRQ_2 = DefaultHandler);
PROVIDE(TIMER1_IRQ_3 = DefaultHandler);
PROVIDE(PWM_IRQ_WRAP_0 = DefaultHandler);
PROVIDE(PWM_IRQ_WRAP_1 = DefaultHandler);
PROVIDE(DMA_IRQ_0 = DefaultHandler);
PROVIDE(DMA_IRQ_1 = DefaultHandler);
PROVIDE(DMA_IRQ_2 = DefaultHandler);
PROVIDE(DMA_IRQ_3 = DefaultHandler);
PROVIDE(USBCTRL_IRQ = DefaultHandler);
PROVIDE(PIO0_IRQ_0 = DefaultHandler);
PROVIDE(PIO0_IRQ_1 = DefaultHandler);
PROVIDE(PIO1_IRQ_0 = DefaultHandler);
PROVIDE(PIO1_IRQ_1 = DefaultHandler);
PROVIDE(PIO2_IRQ_0 = DefaultHandler);
PROVIDE(PIO2_IRQ_1 = DefaultHandler);
PROVIDE(IO_IRQ_BANK0 = DefaultHandler);
PROVIDE(IO_IRQ_BANK0_NS = DefaultHandler);
PROVIDE(IO_IRQ_QSPI = DefaultHandler);
PROVIDE(IO_IRQ_QSPI_NS = DefaultHandler);
PROVIDE(SIO_IRQ_FIFO = DefaultHandler);
PROVIDE(SIO_IRQ_BELL = DefaultHandler);
PROVIDE(SIO_IRQ_FIFO_NS = DefaultHandler);
PROVIDE(SIO_IRQ_BELL_NS = DefaultHandler);
PROVIDE(SIO_IRQ_MTIMECMP = DefaultHandler);
PROVIDE(CLOCKS_IRQ = DefaultHandler);
PROVIDE(SPI0_IRQ = DefaultHandler);
PROVIDE(SPI1_IRQ = DefaultHandler);
PROVIDE(UART0_IRQ = DefaultHandler);
PROVIDE(UART1_IRQ = DefaultHandler);
PROVIDE(ADC_IRQ_FIFO = DefaultHandler);
PROVIDE(I2C0_IRQ = DefaultHandler);
PROVIDE(I2C1_IRQ = DefaultHandler);
PROVIDE(OTP_IRQ = DefaultHandler);
PROVIDE(TRNG_IRQ = DefaultHandler);
PROVIDE(PLL_SYS_IRQ = DefaultHandler);
PROVIDE(PLL_USB_IRQ = DefaultHandler);
PROVIDE(POWMAN_IRQ_POW = DefaultHandler);
PROVIDE(POWMAN_IRQ_TIMER = DefaultHandler);
PROVIDE(SW0_IRQ = DefaultHandler);
PROVIDE(SW1_IRQ = DefaultHandler);
PROVIDE(SW2_IRQ = DefaultHandler);
PROVIDE(SW3_IRQ = DefaultHandler);
PROVIDE(SW4_IRQ = DefaultHandler);
PROVIDE(SW5_IRQ = DefaultHandler);

PROVIDE(DefaultIrqHandler = DefaultInterruptHandler);
PROVIDE(_mp_hook = default_mp_hook);
PROVIDE(_start_trap = default_start_trap);

SECTIONS
{
  .text.dummy (NOLOAD) :
  {
    /* This section is intended to make _stext address work. */
    . = ABSOLUTE(_stext);
  } > FLASH

  .text _stext :
  {
    /* Put reset code first so the image entry is at flash origin. */
    KEEP(*(.init));
    KEEP(*(.init.rust));
    . = ALIGN(4);

    /* RP2350 boot ROM image definition. Must be in the first 4 KiB. */
    __start_block_addr = .;
    KEEP(*(.start_block));
    KEEP(*(.boot_info));
    . = ALIGN(4);

    *(.trap);
    *(.trap.rust);
    *(.text.abort);
    *(.text .text.*);
    . = ALIGN(4);
  } > FLASH

  .bi_entries : ALIGN(4)
  {
    __bi_entries_start = .;
    KEEP(*(.bi_entries));
    . = ALIGN(4);
    __bi_entries_end = .;
  } > FLASH

  .rodata : ALIGN(4)
  {
    *(.srodata .srodata.*);
    *(.rodata .rodata.*);
    . = ALIGN(4);
  } > FLASH

  .data : ALIGN(32)
  {
    _sidata = LOADADDR(.data);
    __sidata = LOADADDR(.data);
    _sdata = .;
    __sdata = .;
    PROVIDE(__global_pointer$ = . + 0x800);
    *(.sdata .sdata.* .sdata2 .sdata2.*);
    *(.data .data.*);
    . = ALIGN(32);
    _edata = .;
    __edata = .;
  } > RAM AT > FLASH

  .bss (NOLOAD) : ALIGN(32)
  {
    _sbss = .;
    *(.sbss .sbss.* .bss .bss.*);
    . = ALIGN(32);
    _ebss = .;
  } > RAM

  /* defmt-rtt stores its ring buffer in .uninit.*. Keep it RAM-only so
   * elf2uf2 does not treat the buffer as initialized flash contents. */
  .uninit (NOLOAD) : ALIGN(4)
  {
    *(.uninit .uninit.*);
    . = ALIGN(4);
  } > RAM

  .end_block : ALIGN(4)
  {
    __end_block_addr = .;
    KEEP(*(.end_block));
    __flash_binary_end = .;
  } > FLASH

  .heap (NOLOAD) :
  {
    _sheap = .;
    . += _heap_size;
    . = ALIGN(4);
    _eheap = .;
  } > RAM

  .stack (NOLOAD) :
  {
    _estack = .;
    . = ABSOLUTE(_stack_start);
    _sstack = .;
  } > RAM

  .got (INFO) :
  {
    KEEP(*(.got .got.*));
  }

  .eh_frame (INFO) : { KEEP(*(.eh_frame)) }
  .eh_frame_hdr (INFO) : { *(.eh_frame_hdr) }
}

PROVIDE(start_to_end = __end_block_addr - __start_block_addr);
PROVIDE(end_to_start = __start_block_addr - __end_block_addr);

ASSERT(ORIGIN(FLASH) % 4 == 0, "
ERROR(riscv-rt): the start of FLASH must be 4-byte aligned");

ASSERT(ORIGIN(RAM) % 32 == 0, "
ERROR(riscv-rt): the start of RAM must be 32-byte aligned");

ASSERT(_stext % 4 == 0, "
ERROR(riscv-rt): _stext must be 4-byte aligned");

ASSERT(_sdata % 32 == 0 && _edata % 32 == 0, "
BUG(riscv-rt): .data is not 32-byte aligned");

ASSERT(_sidata % 32 == 0, "
BUG(riscv-rt): the LMA of .data is not 32-byte aligned");

ASSERT(_sbss % 32 == 0 && _ebss % 32 == 0, "
BUG(riscv-rt): .bss is not 32-byte aligned");

ASSERT(_sheap % 4 == 0, "
BUG(riscv-rt): start of .heap is not 4-byte aligned");

ASSERT(_stext + SIZEOF(.text) < ORIGIN(FLASH) + LENGTH(FLASH), "
ERROR(riscv-rt): .text must be placed inside FLASH");

ASSERT(SIZEOF(.stack) >= (_max_hart_id + 1) * _hart_stack_size, "
ERROR(riscv-rt): .stack section is too small for all harts");

ASSERT(SIZEOF(.got) == 0, "
.got section detected in the input files. Dynamic relocations are unsupported.");
