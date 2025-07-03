use std::{cell::Ref, hash::Hash};

use super::{Value, operation::Operation};

pub type BackwardFn = fn(value: &Ref<ValueInternal>);

#[derive(Debug)]
pub struct ValueInternal {
    pub data: f64,
    pub gradient: f64,
    pub previous: Vec<Value>,
    pub label: Option<String>,
    pub operation: Option<Operation>,
    pub backward: Option<BackwardFn>,
}

impl ValueInternal {
    pub fn new(
        data: f64,
        label: Option<String>,
        operation: Option<Operation>,
        previous: Vec<Value>,
        backward: Option<BackwardFn>,
    ) -> Self {
        Self {
            data,
            gradient: 0.0,
            label,
            operation,
            previous,
            backward,
        }
    }
}

impl Hash for ValueInternal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.gradient.to_bits().hash(state);
        self.previous.hash(state);
        self.label.hash(state);
        self.operation.hash(state);
    }
}

impl PartialEq for ValueInternal {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.gradient == other.gradient
            && self.previous == other.previous
            && self.label == other.label
            && self.operation == other.operation
    }
}

impl Eq for ValueInternal {}
