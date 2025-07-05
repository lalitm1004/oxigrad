use std::ops::Neg;

use crate::value::{BackwardFn, Operation, Value, ValueInternal};

use super::Loss;

impl Loss {
    pub fn stable_softmax(logits: &[f64]) -> Vec<f64> {
        let max = logits.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let exp: Vec<f64> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum_exps: f64 = exp.iter().sum();
        exp.iter().map(|x| x / sum_exps).collect()
    }

    pub fn cross_entropy_helper(logits: &[Value], targets: &[Value]) -> Value {
        let n = logits.len();
        assert_eq!(n, targets.len(), "Logits and Targets must have same length");

        let logits_data: Vec<f64> = logits.iter().map(|v| v.borrow().data).collect();
        let probs = Self::stable_softmax(&logits_data);

        let loss_val = targets
            .iter()
            .enumerate()
            .map(|(i, t)| t.borrow().data * probs[i].ln())
            .sum::<f64>()
            .neg();

        let backward: BackwardFn = |out| {
            let n = match out.operation {
                Some(Operation::CROSSENTROPY(n)) => n,
                _ => panic!("Missing CROSSENTROPY operation in backward"),
            };

            let logits = &out.previous[0..n];
            let targets = &out.previous[n..2 * n];

            let logits_data: Vec<f64> = logits.iter().map(|v| v.borrow().data).collect();
            let probs = Self::stable_softmax(&logits_data);
            let out_gradient = out.gradient;

            for i in 0..n {
                let mut logit = logits[i].borrow_mut();
                let target_val = targets[i].borrow().data;
                logit.gradient += (probs[i] - target_val) * out_gradient;
            }

            for i in 0..n {
                let mut target = targets[i].borrow_mut();
                target.gradient += -probs[i].ln() * out_gradient;
            }
        };

        let mut all_inputs = Vec::with_capacity(2 * n);
        all_inputs.extend_from_slice(logits);
        all_inputs.extend_from_slice(targets);

        Value::new_internal(ValueInternal::new(
            loss_val,
            None,
            Some(Operation::CROSSENTROPY(n)),
            all_inputs,
            Some(backward),
        ))
    }
}

#[cfg(test)]
mod test;
