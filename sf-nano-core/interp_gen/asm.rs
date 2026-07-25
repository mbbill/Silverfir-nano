//! Assembly text sink for the build-time handler generator.
//!
//! The generator emits assembler source, not machine code. Two reasons:
//! the assembler resolves branch labels, which removes the hand-counted
//! branch delta that was this emitter's most frequent bring-up defect; and
//! the result lands in the binary's own `.text` through `global_asm!`, so
//! the interpreter needs no executable memory at run time.
//!
//! Only the syntax that actually differs between the three object formats
//! lives here: the symbol prefix, the local-label prefix, the
//! non-preemptible-visibility directive, and (on Thumb) the interworking
//! bit that a code address carried in data must set.

/// Object-format flavour, derived from the target triple in `build.rs`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ObjFmt {
    Elf,
    MachO,
    Coff,
}

pub struct Asm {
    out: String,
    fmt: ObjFmt,
    /// Set for Thumb-only targets: a code address that travels through a
    /// register must have bit 0 set or the core switches to A32 and dies.
    thumb: bool,
    seq: u32,
    /// Label the whole blob is measured from.
    base: String,
}

impl Asm {
    pub fn new(fmt: ObjFmt, thumb: bool) -> Asm {
        let mut a = Asm {
            out: String::new(),
            fmt,
            thumb,
            seq: 0,
            base: String::new(),
        };
        a.base = a.local("sf_base");
        a
    }

    /// Public symbol name with the platform's mangling.
    pub fn sym(&self, name: &str) -> String {
        match self.fmt {
            ObjFmt::MachO => format!("_{name}"),
            _ => name.to_string(),
        }
    }

    /// An assembler-local label: never emitted into the symbol table, so
    /// ~10 k of them cost nothing in the linked binary.
    pub fn local(&self, name: &str) -> String {
        match self.fmt {
            ObjFmt::MachO => format!("L{name}"),
            _ => format!(".L{name}"),
        }
    }

    /// A fresh local label, for the inside of one handler.
    pub fn fresh(&mut self, tag: &str) -> String {
        self.seq += 1;
        let n = self.seq;
        self.local(&format!("{tag}_{n}"))
    }

    /// One line of output. Everything below funnels through it; writing
    /// into a `String` cannot fail, so nothing here returns a result.
    fn line(&mut self, text: &str) {
        self.out.push_str(text);
        self.out.push('\n');
    }

    pub fn raw(&mut self, text: &str) {
        self.line(text);
    }

    /// One instruction, indented.
    pub fn ins(&mut self, text: &str) {
        self.out.push('\t');
        self.line(text);
    }

    pub fn label(&mut self, name: &str) {
        self.line(&format!("{name}:"));
    }

    pub fn comment(&mut self, text: &str) {
        // `//` is accepted by the integrated assembler on every target we
        // emit for; `#` is an arm32 immediate prefix and `;` ends a
        // statement on some, so neither is portable.
        self.line(&format!("\t// {text}"));
    }

    pub fn align(&mut self, p2: u32) {
        self.line(&format!("\t.p2align {p2}"));
    }

    /// Declare and define a globally visible, non-preemptible symbol.
    pub fn global_label(&mut self, name: &str) {
        let s = self.sym(name);
        self.line(&format!("\t.globl {s}"));
        match self.fmt {
            ObjFmt::MachO => self.line(&format!("\t.private_extern {s}")),
            ObjFmt::Elf => self.line(&format!("\t.hidden {s}")),
            ObjFmt::Coff => {}
        }
        self.line(&format!("{s}:"));
    }

    pub fn base_label(&self) -> String {
        self.base.clone()
    }

    /// Switch to the read-only data section. The handler and meta tables
    /// live here rather than in `.text`: the runtime reads them and
    /// nothing in them is executable.
    pub fn rodata(&mut self) {
        let d = match self.fmt {
            ObjFmt::Elf => "\t.section .rodata",
            ObjFmt::MachO => "\t.section __TEXT,__const",
            ObjFmt::Coff => "\t.section .rdata,\"dr\"",
        };
        self.line(d);
    }

    /// A 32-bit table entry holding a *code* offset from the blob base.
    /// On Thumb the interworking bit rides in it, so the runtime can jump
    /// to `base + entry` with no per-target knowledge.
    pub fn code_off(&mut self, label: &str) {
        let base = self.base.clone();
        let bit = if self.thumb { " + 1" } else { "" };
        self.line(&format!("\t.long {label} - {base}{bit}"));
    }

    /// A 32-bit table entry holding a plain byte distance (a size).
    pub fn plain_off(&mut self, label: &str) {
        let base = self.base.clone();
        self.line(&format!("\t.long {label} - {base}"));
    }

    pub fn word(&mut self, v: u32) {
        self.line(&format!("\t.long {v}"));
    }

    /// The finished source, with braces escaped.
    ///
    /// `global_asm!` parses its argument as a format template, so an ARM
    /// register list (`push {r4-r12, lr}`) is a syntax error until its
    /// braces are doubled. Doing it here rather than in the one backend
    /// that needs it keeps the next emitter from rediscovering it.
    pub fn finish(self) -> String {
        self.out.replace('{', "{{").replace('}', "}}")
    }
}
