//! The engine an embedder instantiates modules in, and the tier it runs.
//!
//! [`Engine`] holds a validated [`Config`] and hands a copy to every
//! instance it creates. It is a value, not a process-wide setting: two
//! engines can run different tiers with different budgets side by side,
//! and neither can be disturbed by the other being configured.
//!
//! [`Tier`]'s variants are gated on the engine features, so a build with a
//! single engine gets a *univariant* enum: it occupies no bytes, carries no
//! discriminant, and every match on it folds to its one arm. The switch is
//! not merely cheap in that build — it is gone. Naming a tier this build
//! does not have is a compile error at the embedder's call site rather than
//! a runtime `Err`.

use crate::config::{Config, ConfigError};

/// An execution engine compiled into this build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Compile each function to native code before running it.
    #[cfg(sf_jit)]
    Jit = 0,
    /// Predecode each function and run it through the threaded dispatch
    /// chain generated for this target at build time.
    #[cfg(sf_interp)]
    Interp = 1,
}

impl Tier {
    /// Used when the embedder expresses no preference: the JIT where it is
    /// compiled in, the interpreter otherwise.
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

    /// Every tier this build has, in preference order.
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

    #[cfg(sf_jit)]
    #[inline]
    pub const fn is_jit(self) -> bool {
        matches!(self, Self::Jit)
    }

    /// Resolve an embedder-facing tier name.
    ///
    /// `None` covers both an unknown name and a tier this build left out;
    /// [`Tier::ALL`] is what a caller reports back to say which names
    /// would have worked.
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

/// The environment modules are instantiated in.
#[derive(Clone, Copy, Debug)]
pub struct Engine {
    config: Config,
}

impl Engine {
    /// Build an engine, rejecting a configuration that cannot run.
    ///
    /// This is where a bare-metal embedder finds out it never set a budget
    /// — once, up front, rather than as a trap somewhere inside a guest.
    pub const fn new(config: Config) -> Result<Self, ConfigError> {
        match config.validate() {
            Ok(()) => Ok(Self { config }),
            Err(err) => Err(err),
        }
    }

    /// An engine on this target's defaults.
    ///
    /// Panics on a target that has no defaults (bare metal); use
    /// [`Engine::new`] there, which reports which budget is missing.
    pub fn with_defaults() -> Self {
        match Self::new(Config::new()) {
            Ok(engine) => engine,
            Err(err) => panic!("no default engine configuration on this target: {err}"),
        }
    }

    #[inline]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    #[inline]
    pub const fn tier(&self) -> Tier {
        self.config.get_tier()
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, Engine, Tier};

    /// The claim the tier design rests on: with one engine compiled in,
    /// naming it costs nothing to store and nothing to branch on.
    #[test]
    fn single_engine_build_makes_the_tier_a_zst() {
        let tiers = Tier::ALL.len();
        let size = core::mem::size_of::<Tier>();
        if tiers == 1 {
            assert_eq!(size, 0, "a univariant Tier must not carry a discriminant");
        } else {
            assert_eq!(tiers, 2);
            assert_eq!(size, 1);
        }
    }

    #[test]
    fn parses_only_the_names_this_build_has() {
        assert_eq!(Tier::parse_str("auto"), Some(Tier::DEFAULT));
        assert_eq!(Tier::parse_str("nonsense"), None);
        for &tier in Tier::ALL {
            assert_eq!(Tier::parse_str(tier.as_str()), Some(tier));
        }
    }

    /// Two engines, two budgets, at the same time — which the process-wide
    /// configuration this replaced could not express at all.
    #[test]
    fn engines_carry_their_own_configuration() {
        let small = Engine::new(Config::new().wasm_stack_bytes(4096)).expect("small");
        let large = Engine::new(Config::new().wasm_stack_bytes(1 << 20)).expect("large");
        assert_eq!(small.config().get_wasm_stack_bytes(), 4096);
        assert_eq!(large.config().get_wasm_stack_bytes(), 1 << 20);
    }
}
