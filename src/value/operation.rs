use std::fmt;

#[derive(Debug, Clone)]
pub enum Operation {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    POWER(f64),
    EXP,

    // Activation Functions
    RELU,
    SIGMOID,

    // Loss Criterion
    CROSSENTROPY(usize),
    MSE(usize),
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::ADD => write!(f, "ADD"),
            Operation::SUBTRACT => write!(f, "SUBTRACT"),
            Operation::MULTIPLY => write!(f, "MULTIPLY"),
            Operation::DIVIDE => write!(f, "DIVIDE"),
            Operation::POWER(exp) => write!(f, "POWER({})", exp),
            Operation::EXP => write!(f, "EXP"),
            Operation::RELU => write!(f, "RELU"),
            Operation::SIGMOID => write!(f, "SIGMOID"),
            Operation::CROSSENTROPY(_) => write!(f, "CROSSENTROPY"),
            Operation::MSE(_) => write!(f, "MSE"),
        }
    }
}
