//! Which execution engine runs a module.
//!
//! One concept, chosen by the embedder, resolved once. The variants are
//! gated on the engine features, so a build with a single engine gets a
//! *univariant* enum: it occupies no bytes, carries no discriminant, and
//! every match on it folds to its one arm. The switch is not merely cheap
//! in that build — it is gone.
//!
//! Naming an engine this build does not have is a compile error at the
//! embedder's call site rather than a runtime `Err`, so a single-engine
//! embedder cannot ship code that asks for the engine it left out.

use core::sync::atomic::{AtomicU8, Ordering};

/// An execution engine compiled into this build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// Compile each function to native code before running it.
    #[cfg(sf_jit)]
    Jit = 0,
    /// Predecode each function and run it through the threaded dispatch
    /// chain generated for this target at build time.
    #[cfg(sf_interp)]
    Interp = 1,
}

impl Engine {
    /// The engine used when the embedder expresses no preference: the JIT
    /// where it is compiled in, the interpreter otherwise.
    pub const DEFAULT: Self = {
        #[cfg(sf_jit)]
        {
            Self::Jit
        }
        #[cfg(not(sf_jit))]
        {
            Self::Interp
        }
    };

    /// Every engine this build has, in preference order.
    pub const ALL: &'static [Self] = &[
        #[cfg(sf_jit)]
        Self::Jit,
        #[cfg(sf_interp)]
        Self::Interp,
    ];

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            #[cfg(sf_jit)]
            Self::Jit => "jit",
            #[cfg(sf_interp)]
            Self::Interp => "interp",
        }
    }

    /// Resolve an embedder-facing engine name.
    ///
    /// `None` covers both an unknown name and an engine this build left
    /// out; [`Engine::ALL`] is what a caller reports back to say which
    /// names would have worked.
    #[inline]
    pub fn parse_str(name: &str) -> Option<Self> {
        match name {
            "auto" => Some(Self::DEFAULT),
            #[cfg(sf_jit)]
            "jit" | "native" => Some(Self::Jit),
            #[cfg(sf_interp)]
            "interp" | "interpreter" => Some(Self::Interp),
            _ => None,
        }
    }
}

static ACTIVE_ENGINE: AtomicU8 = AtomicU8::new(Engine::DEFAULT as u8);

/// Select the engine subsequent instances run on.
pub fn set_engine(engine: Engine) {
    ACTIVE_ENGINE.store(engine as u8, Ordering::Relaxed);
}

/// The engine currently selected.
///
/// In a single-engine build the return type is a ZST, so the load and the
/// mapping below fold away entirely.
pub fn engine() -> Engine {
    match ACTIVE_ENGINE.load(Ordering::Relaxed) {
        #[cfg(sf_jit)]
        x if x == Engine::Jit as u8 => Engine::Jit,
        #[cfg(sf_interp)]
        x if x == Engine::Interp as u8 => Engine::Interp,
        _ => Engine::DEFAULT,
    }
}

#[cfg(test)]
mod tests {
    use super::Engine;

    /// The claim the whole design rests on: with one engine compiled in,
    /// the selector costs nothing to store and nothing to branch on.
    #[test]
    fn single_engine_build_makes_the_selector_a_zst() {
        let engines = Engine::ALL.len();
        let size = core::mem::size_of::<Engine>();
        if engines == 1 {
            assert_eq!(size, 0, "a univariant Engine must not carry a discriminant");
        } else {
            assert_eq!(engines, 2);
            assert_eq!(size, 1);
        }
    }

    #[test]
    fn round_trips_through_the_atomic() {
        for &engine in Engine::ALL {
            super::set_engine(engine);
            assert_eq!(super::engine(), engine);
        }
        super::set_engine(Engine::DEFAULT);
    }

    #[test]
    fn parses_only_the_names_this_build_has() {
        assert_eq!(Engine::parse_str("auto"), Some(Engine::DEFAULT));
        assert_eq!(Engine::parse_str("nonsense"), None);
        for &engine in Engine::ALL {
            assert_eq!(Engine::parse_str(engine.as_str()), Some(engine));
        }
    }
}
