use super::*;
use crate::value::Value;

fn approx_eq(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() < tolerance
}

fn value_with_label(data: f64, label: &str) -> Value {
    Value::new(data, Some(label.to_string()))
}

#[test]
fn test_cross_entropy_basic() {
    // Test basic cross-entropy calculation
    // For a simple 2-class case where target is [1, 0] and logits are [2, 1]
    let logits = vec![
        value_with_label(2.0, "logit_0"),
        value_with_label(1.0, "logit_1"),
    ];
    let targets = vec![
        value_with_label(1.0, "target_0"),
        value_with_label(0.0, "target_1"),
    ];

    let loss = Loss::cross_entropy(logits, targets).unwrap();

    // Manual calculation:
    // softmax([2, 1]) = [e^2/(e^2+e^1), e^1/(e^2+e^1)] ~= [0.731, 0.269]
    // cross_entropy = -(1*ln(0.731) + 0*ln(0.269)) ~= 0.313
    let expected = 0.31326168751822286;
    assert!(approx_eq(loss.borrow().data, expected, 1e-10));
}

#[test]
fn test_cross_entropy_perfect_prediction() {
    // Test case where model perfectly predicts the target
    let logits = vec![
        value_with_label(10.0, "logit_0"),
        value_with_label(0.0, "logit_1"),
    ];
    let targets = vec![
        value_with_label(1.0, "target_0"),
        value_with_label(0.0, "target_1"),
    ];

    let loss = Loss::cross_entropy(logits, targets).unwrap();

    // With very high confidence in correct class, loss should be very small
    assert!(loss.borrow().data < 1e-4);
}

#[test]
fn test_cross_entropy_worst_prediction() {
    // Test case where model predicts opposite of target
    let logits = vec![
        value_with_label(0.0, "logit_0"),
        value_with_label(10.0, "logit_1"),
    ];
    let targets = vec![
        value_with_label(1.0, "target_0"),
        value_with_label(0.0, "target_1"),
    ];

    let loss = Loss::cross_entropy(logits, targets).unwrap();

    // With high confidence in wrong class, loss should be large
    assert!(loss.borrow().data > 10.0);
}

#[test]
fn test_cross_entropy_uniform_distribution() {
    // Test with uniform logits and targets
    let logits = vec![
        value_with_label(0.0, "logit_0"),
        value_with_label(0.0, "logit_1"),
        value_with_label(0.0, "logit_2"),
    ];
    let targets = vec![
        value_with_label(1.0 / 3.0, "target_0"),
        value_with_label(1.0 / 3.0, "target_1"),
        value_with_label(1.0 / 3.0, "target_2"),
    ];

    let loss = Loss::cross_entropy(logits, targets).unwrap();

    // For uniform distribution, loss should be ln(3) ~= 1.099
    let expected = 3.0_f64.ln();
    assert!(approx_eq(loss.borrow().data, expected, 1e-10));
}

#[test]
fn test_cross_entropy_single_class() {
    // Test single class case
    let logits = vec![value_with_label(1.5, "logit_0")];
    let targets = vec![value_with_label(1.0, "target_0")];

    let loss = Loss::cross_entropy(logits, targets).unwrap();

    // For single class, softmax is always 1.0, so loss should be 0
    assert!(approx_eq(loss.borrow().data, 0.0, 1e-10));
}

#[test]
fn test_cross_entropy_multi_class() {
    // Test 4-class classification
    let logits = vec![
        value_with_label(2.0, "logit_0"),
        value_with_label(1.0, "logit_1"),
        value_with_label(0.5, "logit_2"),
        value_with_label(0.1, "logit_3"),
    ];
    let targets = vec![
        value_with_label(0.0, "target_0"),
        value_with_label(1.0, "target_1"),
        value_with_label(0.0, "target_2"),
        value_with_label(0.0, "target_3"),
    ];

    let loss = Loss::cross_entropy(logits, targets).unwrap();

    // Loss should be positive
    assert!(loss.borrow().data > 0.0);
}

#[test]
fn test_cross_entropy_gradient_computation() {
    let logits = vec![
        value_with_label(1.0, "logit_0"),
        value_with_label(2.0, "logit_1"),
    ];
    let targets = vec![
        value_with_label(1.0, "target_0"),
        value_with_label(0.0, "target_1"),
    ];

    let loss = Loss::cross_entropy(logits.clone(), targets.clone()).unwrap();

    loss.backward();

    // Check that gradients were computed
    assert_ne!(logits[0].borrow().gradient, 0.0);
    assert_ne!(logits[1].borrow().gradient, 0.0);
    assert_ne!(targets[0].borrow().gradient, 0.0);
    assert_ne!(targets[1].borrow().gradient, 0.0);
}

#[test]
fn test_cross_entropy_gradient_values() {
    let logits = vec![
        value_with_label(0.0, "logit_0"),
        value_with_label(0.0, "logit_1"),
    ];
    let targets = vec![
        value_with_label(1.0, "target_0"),
        value_with_label(0.0, "target_1"),
    ];

    let loss = Loss::cross_entropy(logits.clone(), targets.clone()).unwrap();

    loss.backward();

    // For uniform logits [0, 0], softmax is [0.5, 0.5]
    // Gradient for logits: (probs - targets) = [0.5-1, 0.5-0] = [-0.5, 0.5]
    assert!(approx_eq(logits[0].borrow().gradient, -0.5, 1e-10));
    assert!(approx_eq(logits[1].borrow().gradient, 0.5, 1e-10));

    // Gradient for targets: -ln(probs) = [-ln(0.5), -ln(0.5)]
    let expected_target_grad = -0.5_f64.ln();
    assert!(approx_eq(
        targets[0].borrow().gradient,
        expected_target_grad,
        1e-10
    ));
    assert!(approx_eq(
        targets[1].borrow().gradient,
        expected_target_grad,
        1e-10
    ));
}

#[test]
fn test_cross_entropy_numerical_stability() {
    let logits = vec![
        value_with_label(100.0, "logit_0"),
        value_with_label(99.0, "logit_1"),
        value_with_label(101.0, "logit_2"),
    ];
    let targets = vec![
        value_with_label(0.0, "target_0"),
        value_with_label(0.0, "target_1"),
        value_with_label(1.0, "target_2"),
    ];

    let loss = Loss::cross_entropy(logits, targets).unwrap();

    // Should not panic or produce NaN/Inf
    assert!(loss.borrow().data.is_finite());
    assert!(loss.borrow().data >= 0.0);
}

#[test]
fn test_cross_entropy_negative_logits() {
    // Test with negative logits
    let logits = vec![
        value_with_label(-2.0, "logit_0"),
        value_with_label(-1.0, "logit_1"),
        value_with_label(-3.0, "logit_2"),
    ];
    let targets = vec![
        value_with_label(0.0, "target_0"),
        value_with_label(1.0, "target_1"),
        value_with_label(0.0, "target_2"),
    ];

    let loss = Loss::cross_entropy(logits, targets).unwrap();

    // Should produce valid loss
    assert!(loss.borrow().data.is_finite());
    assert!(loss.borrow().data >= 0.0);
}

#[test]
fn test_cross_entropy_zero_target() {
    let logits = vec![
        value_with_label(1.0, "logit_0"),
        value_with_label(2.0, "logit_1"),
    ];
    let targets = vec![
        value_with_label(0.0, "target_0"),
        value_with_label(0.0, "target_1"),
    ];

    let loss = Loss::cross_entropy(logits, targets).unwrap();

    // Loss should be 0 when all targets are 0
    assert!(approx_eq(loss.borrow().data, 0.0, 1e-10));
}

#[test]
fn test_cross_entropy_mismatched_lengths() {
    let logits = vec![
        value_with_label(1.0, "logit_0"),
        value_with_label(2.0, "logit_1"),
    ];
    let targets = vec![value_with_label(1.0, "target_0")];

    let result = Loss::cross_entropy(logits, targets);
    assert!(result.is_err());
}

#[test]
fn test_cross_entropy_empty_inputs() {
    // Test error handling for empty inputs
    let logits = vec![];
    let targets = vec![];

    let result = Loss::cross_entropy(logits, targets);
    // Should either work (with 0 loss) or return an error
    if let Ok(loss) = result {
        assert!(approx_eq(loss.borrow().data, 0.0, 1e-10));
    }
}

#[test]
fn test_stable_softmax_overflow_protection() {
    let large_values = vec![700.0, 800.0, 900.0]; // These would overflow regular softmax
    let probs = Loss::stable_softmax(&large_values);

    // Should sum to 1.0 and not contain NaN/Inf
    let sum: f64 = probs.iter().sum();
    assert!(approx_eq(sum, 1.0, 1e-10));
    assert!(probs.iter().all(|&p| p.is_finite() && p >= 0.0));
}

#[test]
fn test_stable_softmax_underflow_protection() {
    // Test with very negative values
    let small_values = vec![-700.0, -800.0, -900.0];
    let probs = Loss::stable_softmax(&small_values);

    // Should sum to 1.0 and not contain NaN/Inf
    let sum: f64 = probs.iter().sum();
    assert!(approx_eq(sum, 1.0, 1e-10));
    assert!(probs.iter().all(|&p| p.is_finite() && p >= 0.0));
}

#[test]
fn test_cross_entropy_symmetry() {
    // Test that cross-entropy is not symmetric (CE(p,q) != CE(q,p))
    let logits = vec![
        value_with_label(2.0, "logit_0"),
        value_with_label(1.0, "logit_1"),
    ];
    let targets = vec![
        value_with_label(1.0, "target_0"),
        value_with_label(0.0, "target_1"),
    ];

    let loss1 = Loss::cross_entropy(logits.clone(), targets.clone()).unwrap();
    let loss2 = Loss::cross_entropy(targets, logits).unwrap();

    assert!(!approx_eq(loss1.borrow().data, loss2.borrow().data, 1e-10));
}

#[test]
fn test_cross_entropy_soft_targets() {
    // Test with soft targets (label smoothing scenario)
    let logits = vec![
        value_with_label(2.0, "logit_0"),
        value_with_label(1.0, "logit_1"),
        value_with_label(0.5, "logit_2"),
    ];
    let targets = vec![
        value_with_label(0.7, "target_0"),
        value_with_label(0.2, "target_1"),
        value_with_label(0.1, "target_2"),
    ];

    let loss = Loss::cross_entropy(logits, targets).unwrap();

    // Should produce reasonable loss value
    assert!(loss.borrow().data.is_finite());
    assert!(loss.borrow().data >= 0.0);
}
