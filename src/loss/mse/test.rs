use super::*;
use crate::value::Value;

fn approx_eq(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() < tolerance
}

fn value_with_label(data: f64, label: &str) -> Value {
    Value::new(data, Some(label.to_string()))
}

#[test]
fn test_mse_basic() {
    // Test basic MSE calculation
    let predictions = vec![
        value_with_label(2.0, "pred_0"),
        value_with_label(1.0, "pred_1"),
    ];
    let targets = vec![
        value_with_label(1.0, "target_0"),
        value_with_label(0.0, "target_1"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    // Manual calculation: ((2-1)^2 + (1-0)^2) / 2 = (1 + 1) / 2 = 1.0
    let expected = 1.0;
    assert!(approx_eq(loss.borrow().data, expected, 1e-10));
}

#[test]
fn test_mse_perfect_prediction() {
    // Test case where predictions exactly match targets
    let predictions = vec![
        value_with_label(1.0, "pred_0"),
        value_with_label(2.0, "pred_1"),
        value_with_label(3.0, "pred_2"),
    ];
    let targets = vec![
        value_with_label(1.0, "target_0"),
        value_with_label(2.0, "target_1"),
        value_with_label(3.0, "target_2"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    // Loss should be exactly 0 for perfect predictions
    assert!(approx_eq(loss.borrow().data, 0.0, 1e-10));
}

#[test]
fn test_mse_large_errors() {
    // Test with large prediction errors
    let predictions = vec![
        value_with_label(10.0, "pred_0"),
        value_with_label(5.0, "pred_1"),
    ];
    let targets = vec![
        value_with_label(0.0, "target_0"),
        value_with_label(0.0, "target_1"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    // Manual calculation: ((10-0)^2 + (5-0)^2) / 2 = (100 + 25) / 2 = 62.5
    let expected = 62.5;
    assert!(approx_eq(loss.borrow().data, expected, 1e-10));
}

#[test]
fn test_mse_negative_values() {
    // Test with negative predictions and targets
    let predictions = vec![
        value_with_label(-2.0, "pred_0"),
        value_with_label(-1.0, "pred_1"),
        value_with_label(1.0, "pred_2"),
    ];
    let targets = vec![
        value_with_label(-1.0, "target_0"),
        value_with_label(-3.0, "target_1"),
        value_with_label(-1.0, "target_2"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    // Manual calculation: ((-2-(-1))^2 + (-1-(-3))^2 + (1-(-1))^2) / 3 = (1 + 4 + 4) / 3 = 3.0
    let expected = 3.0;
    assert!(approx_eq(loss.borrow().data, expected, 1e-10));
}

#[test]
fn test_mse_single_sample() {
    // Test with single prediction-target pair
    let predictions = vec![value_with_label(5.0, "pred_0")];
    let targets = vec![value_with_label(3.0, "target_0")];

    let loss = Loss::mse_helper(&predictions, &targets);

    // Manual calculation: (5-3)^2 / 1 = 4
    let expected = 4.0;
    assert!(approx_eq(loss.borrow().data, expected, 1e-10));
}

#[test]
fn test_mse_fractional_values() {
    // Test with fractional predictions and targets
    let predictions = vec![
        value_with_label(0.5, "pred_0"),
        value_with_label(1.5, "pred_1"),
        value_with_label(2.5, "pred_2"),
    ];
    let targets = vec![
        value_with_label(0.7, "target_0"),
        value_with_label(1.2, "target_1"),
        value_with_label(2.1, "target_2"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    // Manual calculation: ((0.5-0.7)^2 + (1.5-1.2)^2 + (2.5-2.1)^2) / 3 = (0.04 + 0.09 + 0.16) / 3 = 0.29/3
    let expected = 0.29 / 3.0;
    assert!(approx_eq(loss.borrow().data, expected, 1e-10));
}

#[test]
fn test_mse_zero_targets() {
    // Test with all zero targets
    let predictions = vec![
        value_with_label(1.0, "pred_0"),
        value_with_label(2.0, "pred_1"),
        value_with_label(3.0, "pred_2"),
    ];
    let targets = vec![
        value_with_label(0.0, "target_0"),
        value_with_label(0.0, "target_1"),
        value_with_label(0.0, "target_2"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    // Manual calculation: (1^2 + 2^2 + 3^2) / 3 = (1 + 4 + 9) / 3 = 14/3
    let expected = 14.0 / 3.0;
    assert!(approx_eq(loss.borrow().data, expected, 1e-10));
}

#[test]
fn test_mse_zero_predictions() {
    // Test with all zero predictions
    let predictions = vec![
        value_with_label(0.0, "pred_0"),
        value_with_label(0.0, "pred_1"),
        value_with_label(0.0, "pred_2"),
    ];
    let targets = vec![
        value_with_label(1.0, "target_0"),
        value_with_label(2.0, "target_1"),
        value_with_label(3.0, "target_2"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    // Manual calculation: (1^2 + 2^2 + 3^2) / 3 = (1 + 4 + 9) / 3 = 14/3
    let expected = 14.0 / 3.0;
    assert!(approx_eq(loss.borrow().data, expected, 1e-10));
}

#[test]
fn test_mse_gradient_computation() {
    // Test that gradients are computed correctly
    let predictions = vec![
        value_with_label(2.0, "pred_0"),
        value_with_label(1.0, "pred_1"),
    ];
    let targets = vec![
        value_with_label(1.0, "target_0"),
        value_with_label(0.0, "target_1"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    loss.backward();

    // Check that gradients were computed
    assert_ne!(predictions[0].borrow().gradient, 0.0);
    assert_ne!(predictions[1].borrow().gradient, 0.0);
    assert_ne!(targets[0].borrow().gradient, 0.0);
    assert_ne!(targets[1].borrow().gradient, 0.0);
}

#[test]
fn test_mse_gradient_values() {
    // Test exact gradient values
    let predictions = vec![
        value_with_label(3.0, "pred_0"),
        value_with_label(1.0, "pred_1"),
    ];
    let targets = vec![
        value_with_label(1.0, "target_0"),
        value_with_label(2.0, "target_1"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    loss.backward();

    // For MSE, gradient w.r.t. predictions: 2/n * (pred - target)
    // n = 2, so factor is 1.0
    // pred_0 gradient: 1.0 * (3 - 1) = 2.0
    // pred_1 gradient: 1.0 * (1 - 2) = -1.0
    assert!(approx_eq(predictions[0].borrow().gradient, 2.0, 1e-10));
    assert!(approx_eq(predictions[1].borrow().gradient, -1.0, 1e-10));

    // For targets, gradient: 2/n * (target - pred)
    // target_0 gradient: 1.0 * (1 - 3) = -2.0
    // target_1 gradient: 1.0 * (2 - 1) = 1.0
    assert!(approx_eq(targets[0].borrow().gradient, -2.0, 1e-10));
    assert!(approx_eq(targets[1].borrow().gradient, 1.0, 1e-10));
}

#[test]
fn test_mse_gradient_perfect_prediction() {
    // Test gradients when predictions match targets exactly
    let predictions = vec![
        value_with_label(1.0, "pred_0"),
        value_with_label(2.0, "pred_1"),
    ];
    let targets = vec![
        value_with_label(1.0, "target_0"),
        value_with_label(2.0, "target_1"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    loss.backward();

    // Gradients should be zero for perfect predictions
    assert!(approx_eq(predictions[0].borrow().gradient, 0.0, 1e-10));
    assert!(approx_eq(predictions[1].borrow().gradient, 0.0, 1e-10));
    assert!(approx_eq(targets[0].borrow().gradient, 0.0, 1e-10));
    assert!(approx_eq(targets[1].borrow().gradient, 0.0, 1e-10));
}

#[test]
fn test_mse_gradient_single_sample() {
    // Test gradients for single sample
    let predictions = vec![value_with_label(5.0, "pred_0")];
    let targets = vec![value_with_label(3.0, "target_0")];

    let loss = Loss::mse_helper(&predictions, &targets);

    loss.backward();

    // For single sample: gradient factor is 2/1 = 2.0
    // pred gradient: 2.0 * (5 - 3) = 4.0
    // target gradient: 2.0 * (3 - 5) = -4.0
    assert!(approx_eq(predictions[0].borrow().gradient, 4.0, 1e-10));
    assert!(approx_eq(targets[0].borrow().gradient, -4.0, 1e-10));
}

#[test]
fn test_mse_large_dataset() {
    // Test with larger dataset
    let n = 100;
    let predictions: Vec<Value> = (0..n)
        .map(|i| value_with_label(i as f64, &format!("pred_{}", i)))
        .collect();
    let targets: Vec<Value> = (0..n)
        .map(|i| value_with_label((i + 1) as f64, &format!("target_{}", i)))
        .collect();

    let loss = Loss::mse_helper(&predictions, &targets);

    // Each prediction is 1 less than target, so squared error is 1 for each
    // MSE = sum(1^2) / n = n / n = 1.0
    let expected = 1.0;
    assert!(approx_eq(loss.borrow().data, expected, 1e-10));
}

#[test]
fn test_mse_numerical_stability() {
    // Test with very large values
    let predictions = vec![
        value_with_label(1e6, "pred_0"),
        value_with_label(1e6 + 1.0, "pred_1"),
    ];
    let targets = vec![
        value_with_label(1e6 + 0.5, "target_0"),
        value_with_label(1e6 + 1.5, "target_1"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    // Should not produce NaN or Inf
    assert!(loss.borrow().data.is_finite());
    assert!(loss.borrow().data >= 0.0);

    // Manual calculation: ((-0.5)^2 + (-0.5)^2) / 2 = 0.25
    let expected = 0.25;
    assert!(approx_eq(loss.borrow().data, expected, 1e-10));
}

#[test]
fn test_mse_very_small_values() {
    // Test with very small values
    let predictions = vec![
        value_with_label(1e-10, "pred_0"),
        value_with_label(2e-10, "pred_1"),
    ];
    let targets = vec![
        value_with_label(1.5e-10, "target_0"),
        value_with_label(2.5e-10, "target_1"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    // Should handle small values correctly
    assert!(loss.borrow().data.is_finite());
    assert!(loss.borrow().data >= 0.0);
}

#[test]
fn test_mse_symmetric_errors() {
    // Test that MSE treats positive and negative errors equally
    let predictions1 = vec![
        value_with_label(1.0, "pred_0"),
        value_with_label(2.0, "pred_1"),
    ];
    let targets1 = vec![
        value_with_label(3.0, "target_0"),
        value_with_label(4.0, "target_1"),
    ];

    let predictions2 = vec![
        value_with_label(5.0, "pred_0"),
        value_with_label(6.0, "pred_1"),
    ];
    let targets2 = vec![
        value_with_label(3.0, "target_0"),
        value_with_label(4.0, "target_1"),
    ];

    let loss1 = Loss::mse_helper(&predictions1, &targets1);
    let loss2 = Loss::mse_helper(&predictions2, &targets2);

    // Both should have same MSE (errors of magnitude 2)
    assert!(approx_eq(loss1.borrow().data, loss2.borrow().data, 1e-10));
}

#[test]
fn test_mse_regression_case() {
    // Test a realistic regression scenario
    let predictions = vec![
        value_with_label(2.1, "pred_0"),
        value_with_label(1.9, "pred_1"),
        value_with_label(3.05, "pred_2"),
        value_with_label(0.95, "pred_3"),
    ];
    let targets = vec![
        value_with_label(2.0, "target_0"),
        value_with_label(2.0, "target_1"),
        value_with_label(3.0, "target_2"),
        value_with_label(1.0, "target_3"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    // Manual calculation: ((0.1)^2 + (-0.1)^2 + (0.05)^2 + (-0.05)^2) / 4 = (0.01 + 0.01 + 0.0025 + 0.0025) / 4 = 0.025/4 = 0.00625
    let expected = 0.00625;
    assert!(approx_eq(loss.borrow().data, expected, 1e-10));
}

#[test]
#[should_panic(expected = "Predictions and Targets must have same length")]
fn test_mse_mismatched_lengths() {
    // Test error handling for mismatched lengths
    let predictions = vec![
        value_with_label(1.0, "pred_0"),
        value_with_label(2.0, "pred_1"),
    ];
    let targets = vec![value_with_label(1.0, "target_0")];

    Loss::mse_helper(&predictions, &targets);
}

#[test]
fn test_mse_empty_inputs() {
    // Test with empty inputs
    let predictions = vec![];
    let targets = vec![];

    let loss = Loss::mse_helper(&predictions, &targets);

    // MSE of empty set should be 0.0/0.0, but since we divide by n, this would be NaN
    // However, the sum of empty iterator is 0, so we get 0.0/0.0 = NaN
    // This test documents current behavior - might want to handle this case explicitly
    assert!(loss.borrow().data.is_nan());
}

#[test]
fn test_mse_mean_property() {
    // Test that MSE is actually computing the mean
    let predictions = vec![
        value_with_label(1.0, "pred_0"),
        value_with_label(2.0, "pred_1"),
        value_with_label(3.0, "pred_2"),
        value_with_label(4.0, "pred_3"),
    ];
    let targets = vec![
        value_with_label(0.0, "target_0"),
        value_with_label(0.0, "target_1"),
        value_with_label(0.0, "target_2"),
        value_with_label(0.0, "target_3"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    // Manual calculation: (1^2 + 2^2 + 3^2 + 4^2) / 4 = (1 + 4 + 9 + 16) / 4 = 30/4 = 7.5
    let expected = 7.5;
    assert!(approx_eq(loss.borrow().data, expected, 1e-10));
}

#[test]
fn test_mse_gradient_accumulation() {
    // Test that gradients accumulate correctly with multiple backward passes
    let predictions = vec![
        value_with_label(2.0, "pred_0"),
        value_with_label(1.0, "pred_1"),
    ];
    let targets = vec![
        value_with_label(1.0, "target_0"),
        value_with_label(0.0, "target_1"),
    ];

    let loss = Loss::mse_helper(&predictions, &targets);

    loss.backward();
    let first_grad_pred0 = predictions[0].borrow().gradient;
    let first_grad_pred1 = predictions[1].borrow().gradient;

    loss.backward();
    let second_grad_pred0 = predictions[0].borrow().gradient;
    let second_grad_pred1 = predictions[1].borrow().gradient;

    // Gradients should accumulate
    assert!(approx_eq(second_grad_pred0, 2.0 * first_grad_pred0, 1e-10));
    assert!(approx_eq(second_grad_pred1, 2.0 * first_grad_pred1, 1e-10));
}
