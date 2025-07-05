mod cross_entropy;

use pyo3::{exceptions::PyValueError, prelude::*};

use crate::value::Value;

#[pyclass]
pub struct Loss;

#[pymethods]
impl Loss {
    #[staticmethod]
    #[pyo3(name = "CrossEntropy")]
    fn cross_entropy(logits: Vec<Value>, targets: Vec<Value>) -> PyResult<Value> {
        if logits.len() != targets.len() {
            return Err(PyValueError::new_err(
                "Logits and targets must have same length",
            ));
        }
        Ok(Self::cross_entropy_helper(&logits, &targets))
    }
}
