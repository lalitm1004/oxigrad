mod activation;
mod value;

use pyo3::prelude::*;

use activation::Activation;
use value::Value;

#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Value>()?;
    m.add_class::<Activation>()?;
    Ok(())
}
