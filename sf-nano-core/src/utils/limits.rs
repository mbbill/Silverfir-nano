use crate::WasmError;

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    min: usize,
    max: Option<usize>,
    default_max: Option<usize>,
    pub(crate) is64: bool,
}

impl Limits {
    pub fn new(min: usize, max: Option<usize>) -> Result<Self, WasmError> {
        if let Some(max) = max {
            if max < min {
                return Err(WasmError::invalid("min larger than max"));
            }
        }
        Ok(Limits {
            min,
            max,
            default_max: None,
            is64: false,
        })
    }

    pub fn new_64(min: usize, max: Option<usize>) -> Result<Self, WasmError> {
        if let Some(max) = max {
            if max < min {
                return Err(WasmError::invalid("min larger than max"));
            }
        }
        Ok(Limits {
            min,
            max,
            default_max: None,
            is64: true,
        })
    }

    /// Whether these limits describe a 64-bit memory or table.
    pub fn is_64(&self) -> bool {
        self.is64
    }

    /// Returns the minimum value.
    pub fn min(&self) -> usize {
        self.min
    }

    /// Returns the explicit maximum value, if any.
    pub fn max(&self) -> Option<usize> {
        self.max
    }

    /// Returns the effective maximum (explicit max or default max).
    pub(crate) fn effective_max(&self) -> usize {
        self.max
            .unwrap_or_else(|| self.default_max.unwrap_or(usize::MAX))
    }

    pub(crate) fn with_default_max(&self, default_max: usize) -> Result<Self, WasmError> {
        if let Some(max) = self.max {
            if max > default_max {
                return Err(WasmError::invalid("max larger than default max"));
            }
        }
        if self.min > default_max {
            return Err(WasmError::invalid("min larger than default max"));
        }
        Ok(Limits {
            min: self.min,
            max: self.max,
            default_max: Some(default_max),
            is64: self.is64,
        })
    }
}

pub(crate) trait Limitable {
    fn limits(&self) -> &Limits;
}
