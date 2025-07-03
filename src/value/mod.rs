mod internal;
mod operation;

use std::{
    cell::RefCell,
    collections::HashSet,
    hash::Hash,
    ops::{Add, Deref, Mul, Neg, Sub},
    rc::Rc,
};

use internal::{BackwardFn, ValueInternal};
use operation::Operation;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value(Rc<RefCell<ValueInternal>>);

impl Value {
    fn new(value_internal: ValueInternal) -> Self {
        Self(Rc::new(RefCell::new(value_internal)))
    }

    pub fn from<T>(t: T) -> Value
    where
        T: Into<Value>,
    {
        t.into()
    }

    pub fn set_label(self, label: &str) -> Self {
        self.borrow_mut().label = Some(label.to_string());
        self
    }

    pub fn get_label(self) -> Option<String> {
        self.borrow().label.clone()
    }

    pub fn get_data(&self) -> f64 {
        self.borrow().data
    }

    pub fn get_gradient(&self) -> f64 {
        self.borrow().gradient
    }

    pub fn pow(&self, power: &Self) -> Value {
        let result = self.borrow().data.powf(power.borrow().data);

        let backward: BackwardFn = |out| {
            let mut base = out.previous[0].borrow_mut();
            let power = out.previous[1].borrow();

            base.gradient += power.data * (base.data.powf(power.data - 1.0)) * out.gradient;
        };

        Value::new(ValueInternal::new(
            result,
            None,
            Some(Operation::POWER(power.get_data())),
            vec![self.clone(), power.clone()],
            Some(backward),
        ))
    }

    pub fn backprop(&self) {
        let mut visited: HashSet<Value> = HashSet::new();

        self.borrow_mut().gradient = 1.0;
        Self::backprop_helper(&mut visited, self);
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

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, other: Self) -> Self::Output {
        add_using_ref(&self, &other)
    }
}

impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;

    fn add(self, other: &'b Value) -> Self::Output {
        add_using_ref(self, other)
    }
}

fn add_using_ref(a: &Value, b: &Value) -> Value {
    let result = a.borrow().data + b.borrow().data;

    let backward: BackwardFn = |out| {
        let mut first = out.previous[0].borrow_mut();
        let mut second = out.previous[1].borrow_mut();

        first.gradient += out.gradient;
        second.gradient += out.gradient;
    };

    Value::new(ValueInternal::new(
        result,
        None,
        Some(Operation::ADD),
        vec![a.clone(), b.clone()],
        Some(backward),
    ))
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Self) -> Self::Output {
        mul_using_ref(&self, &other)
    }
}

impl<'a, 'b> Mul<&'b Value> for &'a Value {
    type Output = Value;

    fn mul(self, other: &'b Value) -> Self::Output {
        mul_using_ref(self, other)
    }
}

fn mul_using_ref(a: &Value, b: &Value) -> Value {
    let result = a.borrow().data * b.borrow().data;

    let backward: BackwardFn = |out| {
        let mut first = out.previous[0].borrow_mut();
        let mut second = out.previous[1].borrow_mut();

        first.gradient += second.data * out.gradient;
        second.gradient += first.data * out.gradient;
    };

    Value::new(ValueInternal::new(
        result,
        None,
        Some(Operation::MULTIPLY),
        vec![a.clone(), b.clone()],
        Some(backward),
    ))
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        mul_using_ref(&self, &Value::from(-1))
    }
}

impl<'a> Neg for &'a Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        mul_using_ref(self, &Value::from(-1))
    }
}

impl Sub<Value> for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Self::Output {
        add_using_ref(&self, &(-other))
    }
}

impl<'a, 'b> Sub<&'b Value> for &'a Value {
    type Output = Value;

    fn sub(self, other: &'b Value) -> Self::Output {
        add_using_ref(self, &(-other))
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

impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(ValueInternal::new(t.into(), None, None, vec![], None))
    }
}
