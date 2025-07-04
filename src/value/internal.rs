use std::cell::Ref;

use super::{operation::Operation, Value};

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
