use pyo3::prelude::*;

use crate::{
    loss::Loss,
    value::{BackwardFn, Operation, Value, ValueInternal},
};

#[pyclass]
pub struct Activation;

#[pymethods]
impl Activation {
    #[staticmethod]
    #[pyo3(name = "ReLU")]
    pub fn relu(input: &Value) -> Value {
        let input_data = input.borrow().data;

        let result = if input_data > 0.0 { input_data } else { 0.0 };

        let backward: BackwardFn = |out| {
            let mut prev = out.previous[0].borrow_mut();

            prev.gradient += (if out.data > 0.0 { 1.0 } else { 0.0 }) * out.gradient;
        };

        Value::new_internal(ValueInternal::new(
            result,
            None,
            Some(Operation::RELU),
            vec![input.clone()],
            Some(backward),
        ))
    }

    #[staticmethod]
    #[pyo3(name = "Sigmoid")]
    pub fn sigmoid(input: &Value) -> Value {
        let result = 1.0 / (1.0 + input.borrow().data.exp());

        let backward: BackwardFn = |out| {
            let mut prev = out.previous[0].borrow_mut();

            prev.gradient += out.data * (1.0 - out.data)
        };

        Value::new_internal(ValueInternal::new(
            result,
            None,
            Some(Operation::SIGMOID),
            vec![input.clone()],
            Some(backward),
        ))
    }

    #[staticmethod]
    #[pyo3(name = "Softmax")]
    pub fn softmax(logits: Vec<Value>) -> Vec<f64> {
        let logits_data: Vec<f64> = logits.iter().map(|v| v.borrow().data).collect();
        Loss::stable_softmax(&logits_data)
    }
}
