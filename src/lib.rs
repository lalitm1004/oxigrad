use pyo3::prelude::*;

mod internal;
mod operation;

use std::{cell::RefCell, collections::HashSet, hash::Hash, ops::Deref, rc::Rc};

use internal::{BackwardFn, ValueInternal};
use operation::Operation;

#[pyclass(unsendable)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value(Rc<RefCell<ValueInternal>>);

#[pymethods]
impl Value {
    #[new]
    fn new(data: f64) -> Self {
        Self(Rc::new(RefCell::new(ValueInternal::new(
            data,
            None,
            None,
            vec![],
            None,
        ))))
    }

    #[staticmethod]
    fn from_float(data: f64) -> Self {
        Self::new(data)
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

    #[getter]
    fn grad(&self) -> f64 {
        self.borrow().gradient
    }

    fn pow(&self, power: &Self) -> Value {
        let result = self.borrow().data.powf(power.borrow().data);

        let backward: BackwardFn = |out| {
            let mut base = out.previous[0].borrow_mut();
            let power = out.previous[1].borrow();

            base.gradient += power.data * (base.data.powf(power.data - 1.0)) * out.gradient;
        };

        Value::new_internal(ValueInternal::new(
            result,
            None,
            Some(Operation::POWER(power.borrow().data)),
            vec![self.clone(), power.clone()],
            Some(backward),
        ))
    }

    fn backward(&self) {
        let mut visited: HashSet<Value> = HashSet::new();

        self.borrow_mut().gradient = 1.0;
        Self::backprop_helper(&mut visited, self);
    }

    fn __add__(&self, other: &Self) -> Value {
        self.add_ref(other)
    }

    fn __mul__(&self, other: &Self) -> Value {
        self.mul_ref(other)
    }

    fn __sub__(&self, other: &Self) -> Value {
        self.sub_ref(other)
    }

    fn __neg__(&self) -> Value {
        self.neg_ref()
    }

    fn __repr__(&self) -> String {
        let borrowed = self.borrow();
        format!(
            "Value(data={:.4}, grad={:.4}{})",
            borrowed.data,
            borrowed.gradient,
            if let Some(ref label) = borrowed.label {
                format!(", label='{}'", label)
            } else {
                String::new()
            }
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl Value {
    fn new_internal(value_internal: ValueInternal) -> Self {
        Self(Rc::new(RefCell::new(value_internal)))
    }

    fn add_ref(&self, other: &Value) -> Value {
        let result = self.borrow().data + other.borrow().data;

        let backward: BackwardFn = |out| {
            let mut first = out.previous[0].borrow_mut();
            let mut second = out.previous[1].borrow_mut();

            first.gradient += out.gradient;
            second.gradient += out.gradient;
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
            let mut first = out.previous[0].borrow_mut();
            let mut second = out.previous[1].borrow_mut();

            first.gradient += second.data * out.gradient;
            second.gradient += first.data * out.gradient;
        };

        Value::new_internal(ValueInternal::new(
            result,
            None,
            Some(Operation::MULTIPLY),
            vec![self.clone(), other.clone()],
            Some(backward),
        ))
    }

    fn sub_ref(&self, other: &Value) -> Value {
        let neg_other = other.neg_ref();
        self.add_ref(&neg_other)
    }

    fn neg_ref(&self) -> Value {
        let minus_one = Value::new(-1.0);
        self.mul_ref(&minus_one)
    }

    fn backprop_helper(visited: &mut HashSet<Value>, value: &Value) {
        if !visited.contains(value) {
            visited.insert(value.clone());

            let temp = value.borrow();
            if let Some(backward) = temp.backward {
                backward(&temp)
            }

            for prev in &temp.previous {
                Self::backprop_helper(visited, prev);
            }
        }
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.borrow().hash(state);
    }
}

impl Deref for Value {
    type Target = Rc<RefCell<ValueInternal>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Value>()?;
    Ok(())
}
