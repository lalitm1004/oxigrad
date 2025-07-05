mod internal;
mod operation;

pub use internal::{BackwardFn, ValueInternal};
pub use operation::Operation;

use pyo3::{prelude::*, types::PyAny};
use std::{cell::RefCell, collections::HashSet, hash::Hash, ops::Deref, rc::Rc};

#[pyclass(unsendable)]
#[derive(Debug, Clone)]
pub struct Value(Rc<RefCell<ValueInternal>>);

#[pymethods]
impl Value {
    #[new]
    #[pyo3(signature = (data, label=None))]
    pub fn new(data: f64, label: Option<String>) -> Self {
        Self(Rc::new(RefCell::new(ValueInternal::new(
            data,
            label,
            None,
            vec![],
            None,
        ))))
    }

    fn set_label(&self, label: &str) -> Self {
        self.borrow_mut().label = Some(label.to_string());
        self.clone()
    }

    fn get_label(&self) -> Option<String> {
        self.borrow().label.clone()
    }

    #[getter]
    fn data(&self) -> f64 {
        self.borrow().data
    }

    #[setter]
    fn set_data(&self, new_data: f64) {
        self.borrow_mut().data = new_data
    }

    #[getter]
    fn grad(&self) -> f64 {
        self.borrow().gradient
    }

    #[getter]
    fn previous(&self) -> Vec<Value> {
        self.borrow().previous.clone()
    }

    #[getter]
    fn operation(&self) -> Option<String> {
        let borrowed = self.borrow();

        match borrowed.operation.clone() {
            Some(operation) => Some(operation.to_string()),
            None => None,
        }
    }

    fn exp(&self) -> Value {
        let result = self.borrow().data.exp();

        let backward: BackwardFn = |out| {
            let mut base = out.previous[0].borrow_mut();
            base.gradient += base.data.exp() * out.gradient;
        };

        Value::new_internal(ValueInternal::new(
            result,
            None,
            Some(Operation::EXP),
            vec![self.clone()],
            Some(backward),
        ))
    }

    pub fn zero_grad(&self) {
        let mut visited: HashSet<Value> = HashSet::new();
        Self::zero_grad_helper(&mut visited, self);
    }

    pub fn backward(&self) {
        self.zero_grad();

        self.borrow_mut().gradient = 1.0;
        Self::backprop_helper(self);
    }

    fn __add__(&self, other: Py<PyAny>, py: Python) -> PyResult<Value> {
        let other = Self::from_pyany(other, py)?;
        Ok(self.add_ref(&other))
    }

    fn __mul__(&self, other: Py<PyAny>, py: Python) -> PyResult<Value> {
        let other = Self::from_pyany(other, py)?;
        Ok(self.mul_ref(&other))
    }

    fn __pow__(&self, power: f64, _modulo: Option<Py<PyAny>>) -> Value {
        self.pow_f64(power)
    }

    fn __sub__(&self, other: Py<PyAny>, py: Python) -> PyResult<Value> {
        let other = Self::from_pyany(other, py)?;
        Ok(self.sub_ref(&other))
    }

    fn __truediv__(&self, other: Py<PyAny>, py: Python) -> PyResult<Value> {
        let other = Self::from_pyany(other, py)?;
        Ok(self.div_ref(&other))
    }

    fn __neg__(&self) -> Value {
        self.neg_ref()
    }

    fn __radd__(&self, other: Py<PyAny>, py: Python) -> PyResult<Value> {
        let other_value = Self::from_pyany(other, py)?;
        Ok(other_value.add_ref(self))
    }

    fn __rmul__(&self, other: Py<PyAny>, py: Python) -> PyResult<Value> {
        let other_value = Self::from_pyany(other, py)?;
        Ok(other_value.mul_ref(self))
    }

    fn __rsub__(&self, other: Py<PyAny>, py: Python) -> PyResult<Value> {
        let other_value = Self::from_pyany(other, py)?;
        Ok(other_value.sub_ref(self))
    }

    fn __rtruediv__(&self, other: Py<PyAny>, py: Python) -> PyResult<Value> {
        let other_value = Self::from_pyany(other, py)?;
        Ok(other_value.div_ref(self))
    }

    fn __repr__(&self) -> String {
        let borrowed = self.borrow();

        let label_string = if let Some(ref label) = borrowed.label {
            format!(", label='{}'", label)
        } else {
            String::new()
        };

        let operation_string = if let Some(ref operation) = borrowed.operation {
            format!(", operation='{}'", operation.to_string())
        } else {
            String::new()
        };

        format!(
            "Value(data={:.4}, grad={:.4}{}{})",
            borrowed.data, borrowed.gradient, label_string, operation_string
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl Value {
    pub fn new_internal(value_internal: ValueInternal) -> Self {
        Self(Rc::new(RefCell::new(value_internal)))
    }

    fn from_pyany(obj: Py<PyAny>, py: Python) -> PyResult<Value> {
        if let Ok(val) = obj.extract::<Value>(py) {
            Ok(val)
        } else if let Ok(f) = obj.extract::<f64>(py) {
            Ok(Value::new(f, None))
        } else if let Ok(i) = obj.extract::<i64>(py) {
            Ok(Value::new(i as f64, None))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for operation",
            ))
        }
    }

    fn add_ref(&self, other: &Value) -> Value {
        let result = self.borrow().data + other.borrow().data;

        let backward: BackwardFn = |out| {
            // drop to prevent multiple mutable references in b = a + a
            let mut first = out.previous[0].borrow_mut();
            first.gradient += out.gradient;
            std::mem::drop(first);

            let mut second = out.previous[1].borrow_mut();
            second.gradient += out.gradient;
            std::mem::drop(second);
        };

        Value::new_internal(ValueInternal::new(
            result,
            None,
            Some(Operation::ADD),
            vec![self.clone(), other.clone()],
            Some(backward),
        ))
    }

    fn mul_ref(&self, other: &Value) -> Value {
        let result = self.borrow().data * other.borrow().data;

        let backward: BackwardFn = |out| {
            let first_data = out.previous[0].borrow().data;
            let second_data = out.previous[1].borrow().data;

            // prevent multiple mutable references in b = a * a
            let mut first = out.previous[0].borrow_mut();
            first.gradient += second_data * out.gradient;
            std::mem::drop(first);

            let mut second = out.previous[1].borrow_mut();
            second.gradient += first_data * out.gradient;
            std::mem::drop(second)
        };

        Value::new_internal(ValueInternal::new(
            result,
            None,
            Some(Operation::MULTIPLY),
            vec![self.clone(), other.clone()],
            Some(backward),
        ))
    }

    fn pow_f64(&self, power: f64) -> Value {
        let result = self.borrow().data.powf(power);

        let backward: BackwardFn = |out| {
            if let Some(Operation::POWER(power)) = out.operation {
                let mut base = out.previous[0].borrow_mut();

                base.gradient += power * base.data.powf(power - 1.0) * out.gradient;
            };
        };

        Value::new_internal(ValueInternal::new(
            result,
            None,
            Some(Operation::POWER(power)),
            vec![self.clone()],
            Some(backward),
        ))
    }

    fn sub_ref(&self, other: &Value) -> Value {
        let neg_other = other.neg_ref();
        self.add_ref(&neg_other)
    }

    fn neg_ref(&self) -> Value {
        let minus_one = Value::new(-1.0, None);
        self.mul_ref(&minus_one)
    }

    fn div_ref(&self, other: &Value) -> Value {
        let reciprocal = other.pow_f64(-1.0);
        self.mul_ref(&reciprocal)
    }

    fn zero_grad_helper(visited: &mut HashSet<Value>, value: &Value) {
        if !visited.contains(value) {
            visited.insert(value.clone());

            let previous = value.borrow().previous.clone();

            value.borrow_mut().gradient = 0.0;

            for prev in &previous {
                Self::zero_grad_helper(visited, prev);
            }
        }
    }

    fn backprop_helper(value: &Value) {
        let mut topo_order = Vec::new();
        let mut visited = HashSet::new();
        Self::build_topo_order(value, &mut visited, &mut topo_order);

        for node in topo_order.iter().rev() {
            let temp = node.borrow();
            if let Some(backward) = temp.backward {
                backward(&temp);
            }
        }
    }

    fn build_topo_order(value: &Value, visited: &mut HashSet<Value>, topo_order: &mut Vec<Value>) {
        if visited.contains(value) {
            return;
        }

        visited.insert(value.clone());

        let temp = value.borrow();
        for prev in &temp.previous {
            Self::build_topo_order(prev, visited, topo_order);
        }

        topo_order.push(value.clone());
    }
}

// Use pointer based PartialEq and Hashing to ensure
// each computational path is reached
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self, &other)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self).hash(state);
    }
}

impl Deref for Value {
    type Target = Rc<RefCell<ValueInternal>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod test;
