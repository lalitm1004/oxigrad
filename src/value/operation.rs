use std::{fmt, hash::Hash};

#[derive(Debug, Clone)]
pub enum Operation {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    POWER(f64),
    RELU,
    SIGMOID,
}

impl PartialEq for Operation {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Operation::ADD, Operation::ADD) => true,
            (Operation::SUBTRACT, Operation::SUBTRACT) => true,
            (Operation::MULTIPLY, Operation::MULTIPLY) => true,
            (Operation::DIVIDE, Operation::DIVIDE) => true,
            (Operation::POWER(a), Operation::POWER(b)) => a == b,
            (Operation::RELU, Operation::RELU) => true,
            (Operation::SIGMOID, Operation::SIGMOID) => true,
            _ => false,
        }
    }
}

impl Eq for Operation {}

impl Hash for Operation {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Operation::ADD => 0u8.hash(state),
            Operation::SUBTRACT => 1u8.hash(state),
            Operation::MULTIPLY => 2u8.hash(state),
            Operation::DIVIDE => 3u8.hash(state),
            Operation::POWER(val) => {
                4u8.hash(state);
                val.to_bits().hash(state);
            }
            Operation::RELU => 5u8.hash(state),
            Operation::SIGMOID => 6u8.hash(state),
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::ADD => write!(f, "ADD"),
            Operation::SUBTRACT => write!(f, "SUBTRACT"),
            Operation::MULTIPLY => write!(f, "MULTIPLY"),
            Operation::DIVIDE => write!(f, "DIVIDE"),
            Operation::POWER(exp) => write!(f, "POWER({})", exp),
            Operation::RELU => write!(f, "RELU"),
            Operation::SIGMOID => write!(f, "SIGMOID"),
        }
    }
}
