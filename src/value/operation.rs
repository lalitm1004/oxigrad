use std::hash::Hash;

#[derive(Debug)]
pub enum Operation {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    POWER(f64),
}

impl PartialEq for Operation {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Operation::ADD, Operation::ADD) => true,
            (Operation::SUBTRACT, Operation::SUBTRACT) => true,
            (Operation::MULTIPLY, Operation::MULTIPLY) => true,
            (Operation::DIVIDE, Operation::DIVIDE) => true,
            (Operation::POWER(a), Operation::POWER(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Operation {}

impl Hash for Operation {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Operation::ADD => {
                0u8.hash(state);
            }
            Operation::SUBTRACT => {
                1u8.hash(state);
            }
            Operation::MULTIPLY => {
                2u8.hash(state);
            }
            Operation::DIVIDE => {
                3u8.hash(state);
            }
            Operation::POWER(val) => {
                4u8.hash(state);
                val.to_bits().hash(state);
            }
        }
    }
}
