mod activation;
mod loss;
mod value;

use pyo3::prelude::*;

use activation::Activation;
use loss::Loss;
use value::Value;

#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Activation>()?;
    m.add_class::<Loss>()?;
    m.add_class::<Value>()?;
    Ok(())
}
