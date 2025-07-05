use crate::value::{BackwardFn, Operation, Value, ValueInternal};

use super::Loss;

impl Loss {
    pub fn mse_helper(predictions: &[Value], targets: &[Value]) -> Value {
        let n = predictions.len();
        assert_eq!(
            n,
            targets.len(),
            "Predictions and Targets must have same length"
        );

        let loss_val = predictions
            .iter()
            .zip(targets)
            .map(|(p, t)| {
                let p_val = p.borrow().data;
                let t_val = t.borrow().data;
                (p_val - t_val).powi(2)
            })
            .sum::<f64>()
            / n as f64;

        let backward: BackwardFn = |out| {
            let n = match out.operation {
                Some(Operation::MSE(n)) => n,
                _ => panic!("Missing MSE operation in backward"),
            };

            let predictions = &out.previous[0..n];
            let targets = &out.previous[n..2 * n];
            let out_gradient = out.gradient;

            for i in 0..n {
                let mut pred = predictions[i].borrow_mut();
                let target_val = targets[i].borrow().data;
                pred.gradient += (2.0 / n as f64) * (pred.data - target_val) * out_gradient;
            }

            for i in 0..n {
                let mut target = targets[i].borrow_mut();
                let pred_val = predictions[i].borrow().data;
                target.gradient += (2.0 / n as f64) * (target.data - pred_val) * out_gradient;
            }
        };

        let mut all_inputs = Vec::with_capacity(2 * n);
        all_inputs.extend_from_slice(predictions);
        all_inputs.extend_from_slice(targets);

        Value::new_internal(ValueInternal::new(
            loss_val,
            None,
            Some(Operation::MSE(n)),
            all_inputs,
            Some(backward),
        ))
    }
}

#[cfg(test)]
mod test;
