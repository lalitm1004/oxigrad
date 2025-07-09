#[derive(Debug)]
pub enum DType {
    U8,
    U16,
    U32,
    U64,

    I8,
    I16,
    I32,
    I64,

    F16,
    F32,
    F64,
}

#[derive(Debug)]
pub struct DTypeParseError(String);

impl std::fmt::Display for DTypeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unable to parse '{}' as a dtype", self.0)
    }
}

impl std::error::Error for DTypeParseError {}

impl std::str::FromStr for DType {
    type Err = DTypeParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "u8" => Ok(Self::U8),
            "u16" => Ok(Self::U16),
            "u32" => Ok(Self::U32),
            "u64" => Ok(Self::U64),

            "i8" => Ok(Self::I8),
            "i16" => Ok(Self::I16),
            "i32" => Ok(Self::I32),
            "i64" => Ok(Self::I64),

            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),

            _ => Err(DTypeParseError(s.to_string())),
        }
    }
}

impl DType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",

            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",

            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Self::U8 | Self::U32 | Self::I32 | Self::I64)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    pub fn is_unsigned(&self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    pub fn is_signed(&self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }
}
