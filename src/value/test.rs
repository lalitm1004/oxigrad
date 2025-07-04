use super::*;

fn approx_eq(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() < tolerance
}

fn value_with_label(data: f64, label: &str) -> Value {
    Value::new(data, Some(label.to_string()))
}

#[test]
fn test_addition() {
    let a = value_with_label(3.0, "a");
    let b = value_with_label(2.0, "b");

    let c = a.add_ref(&b);
    assert_eq!(c.data(), 5.0);
    assert_eq!(c.operation(), Some(Operation::ADD.to_string()));
    assert_eq!(c.previous().len(), 2);
}

#[test]
fn test_addition_backward() {
    let a = value_with_label(3.0, "a");
    let b = value_with_label(2.0, "b");

    let c = a.add_ref(&b);
    c.backward();

    assert_eq!(a.grad(), 1.0);
    assert_eq!(b.grad(), 1.0);
    assert_eq!(c.grad(), 1.0);
}

#[test]
fn test_multiplication() {
    let a = value_with_label(4.0, "a");
    let b = value_with_label(3.0, "b");

    let c = a.mul_ref(&b);
    assert_eq!(c.data(), 12.0);
    assert_eq!(c.operation(), Some(Operation::MULTIPLY.to_string()));
    assert_eq!(c.previous().len(), 2);
}

#[test]
fn test_multiplication_backward() {
    let a = value_with_label(4.0, "a");
    let b = value_with_label(3.0, "b");

    let c = a.mul_ref(&b);
    c.backward();

    // a.grad() should be b.data() and vice versa
    assert_eq!(a.grad(), 3.0);
    assert_eq!(b.grad(), 4.0);
    assert_eq!(c.grad(), 1.0);
}

#[test]
fn test_power() {
    let a = value_with_label(2.0, "a");
    let c = a.pow_f64(3.0);

    assert_eq!(c.data(), 8.0);
    assert_eq!(c.operation(), Some("POWER(3)".to_string()));
    assert_eq!(c.previous().len(), 1);
}

#[test]
fn test_power_backward() {
    let a = value_with_label(2.0, "a");
    let c = a.pow_f64(3.0);
    c.backward();

    // d/dx(x^3) = 3x^2, so at x=2: 3*2^2 = 12
    assert_eq!(a.grad(), 12.0);
    assert_eq!(c.grad(), 1.0);
}

#[test]
fn test_subtraction() {
    let a = value_with_label(5.0, "a");
    let b = value_with_label(3.0, "b");

    let c = a.sub_ref(&b);
    assert_eq!(c.data(), 2.0);
}

#[test]
fn test_subtraction_backward() {
    let a = value_with_label(5.0, "a");
    let b = value_with_label(3.0, "b");

    let c = a.sub_ref(&b);
    c.backward();

    assert_eq!(a.grad(), 1.0);
    assert_eq!(b.grad(), -1.0); // derivative of -b is -1
}

#[test]
fn test_negation() {
    let a = value_with_label(5.0, "a");
    let c = a.neg_ref();

    assert_eq!(c.data(), -5.0);
}

#[test]
fn test_negation_backward() {
    let a = value_with_label(5.0, "a");
    let c = a.neg_ref();
    c.backward();

    assert_eq!(a.grad(), -1.0);
}

#[test]
fn test_complex_expression() {
    // Test: f(x, y) = x^2 + 2*x*y + y^2
    let x = value_with_label(3.0, "x");
    let y = value_with_label(4.0, "y");

    let x_squared = x.pow_f64(2.0);
    let y_squared = y.pow_f64(2.0);
    let xy = x.mul_ref(&y);
    let two_xy = xy.mul_ref(&Value::new(2.0, None));

    let result = x_squared.add_ref(&two_xy).add_ref(&y_squared);

    // f(3, 4) = 9 + 24 + 16 = 49
    assert_eq!(result.data(), 49.0);

    result.backward();

    // df/dx = 2x + 2y = 2*3 + 2*4 = 14
    assert_eq!(x.grad(), 14.0);
    // df/dy = 2x + 2y = 2*3 + 2*4 = 14
    assert_eq!(y.grad(), 14.0);
}

#[test]
fn test_self_addition() {
    // Test: f(x) = x + x
    let x = value_with_label(5.0, "x");
    let result = x.add_ref(&x);

    assert_eq!(result.data(), 10.0);

    result.backward();

    // df/dx = 1 + 1 = 2
    assert_eq!(x.grad(), 2.0);
}

#[test]
fn test_self_multiplication() {
    // Test: f(x) = x * x
    let x = value_with_label(3.0, "x");
    let result = x.mul_ref(&x);

    assert_eq!(result.data(), 9.0);

    result.backward();

    // df/dx = x + x = 2x = 6
    assert_eq!(x.grad(), 6.0);
}

#[test]
fn test_zero_grad() {
    let x = value_with_label(2.0, "x");
    let y = value_with_label(3.0, "y");
    let z = x.mul_ref(&y);

    z.backward();

    // Check gradients are set
    assert_eq!(x.grad(), 3.0);
    assert_eq!(y.grad(), 2.0);

    // Zero gradients
    z.zero_grad();

    // Check gradients are zeroed
    assert_eq!(x.grad(), 0.0);
    assert_eq!(y.grad(), 0.0);
    assert_eq!(z.grad(), 0.0);
}

#[test]
fn test_multiple_backward_calls() {
    let x = value_with_label(2.0, "x");
    let y = x.pow_f64(2.0);

    // First backward
    y.backward();
    assert_eq!(x.grad(), 4.0);

    // Second backward - should reset gradients first
    y.backward();
    assert_eq!(x.grad(), 4.0);
}

#[test]
fn test_chain_rule() {
    // Test: f(x) = (x^2 + 1)^3
    let x = value_with_label(2.0, "x");
    let x_squared = x.pow_f64(2.0);
    let inner = x_squared.add_ref(&Value::new(1.0, None));
    let result = inner.pow_f64(3.0);

    // f(2) = (4 + 1)^3 = 125
    assert_eq!(result.data(), 125.0);

    result.backward();

    // df/dx = 3(x^2 + 1)^2 * 2x = 6x(x^2 + 1)^2
    // At x=2: 6*2*(4+1)^2 = 12*25 = 300
    assert_eq!(x.grad(), 300.0);
}

#[test]
fn test_power_edge_cases() {
    // Test x^0 = 1
    let x = value_with_label(5.0, "x");
    let result = x.pow_f64(0.0);
    assert_eq!(result.data(), 1.0);

    result.backward();
    // d/dx(x^0) = 0
    assert_eq!(x.grad(), 0.0);

    // Test x^1 = x
    let x = value_with_label(3.0, "x");
    let result = x.pow_f64(1.0);
    assert_eq!(result.data(), 3.0);

    result.backward();
    // d/dx(x^1) = 1
    assert_eq!(x.grad(), 1.0);
}

#[test]
fn test_negative_numbers() {
    let x = value_with_label(-2.0, "x");
    let y = value_with_label(-3.0, "y");

    let result = x.mul_ref(&y);
    assert_eq!(result.data(), 6.0);

    result.backward();
    assert_eq!(x.grad(), -3.0);
    assert_eq!(y.grad(), -2.0);
}

#[test]
fn test_zero_values() {
    let x = value_with_label(0.0, "x");
    let y = value_with_label(5.0, "y");

    let result = x.mul_ref(&y);
    assert_eq!(result.data(), 0.0);

    result.backward();
    assert_eq!(x.grad(), 5.0);
    assert_eq!(y.grad(), 0.0);
}

#[test]
fn test_fractional_power() {
    let x = value_with_label(4.0, "x");
    let result = x.pow_f64(0.5); // square root

    assert_eq!(result.data(), 2.0);

    result.backward();
    // d/dx(x^0.5) = 0.5 * x^(-0.5) = 0.5 / sqrt(x) = 0.5 / 2 = 0.25
    assert!(approx_eq(x.grad(), 0.25, 1e-10));
}

#[test]
fn test_deep_computation_graph() {
    // Test a deeper computation graph
    let x = value_with_label(2.0, "x");
    let mut result = x.clone();

    // Build: ((((x^2)^2)^2)^2) = x^8
    for _ in 0..3 {
        result = result.pow_f64(2.0);
    }

    // x^8 at x=2 should be 256
    assert_eq!(result.data(), 256.0);

    result.backward();

    // d/dx(x^8) = 8x^7 = 8 * 2^7 = 8 * 128 = 1024
    assert_eq!(x.grad(), 1024.0);
}

#[test]
fn test_multiple_paths_to_same_variable() {
    // Test: f(x) = x^2 + x^3
    let x = value_with_label(2.0, "x");
    let x_squared = x.pow_f64(2.0);
    let x_cubed = x.pow_f64(3.0);
    let result = x_squared.add_ref(&x_cubed);

    // f(2) = 4 + 8 = 12
    assert_eq!(result.data(), 12.0);

    result.backward();

    // df/dx = 2x + 3x^2 = 2*2 + 3*4 = 4 + 12 = 16
    assert_eq!(x.grad(), 16.0);
}

#[test]
fn test_large_computation_graph() {
    // Test with many operations to ensure no stack overflow
    let x = value_with_label(1.0, "x");
    let mut result = x.clone();

    // Chain many additions: x + 1 + 1 + 1 + ... (100 times)
    for i in 1..100 {
        let constant = Value::new(1.0, Some(format!("c{}", i)));
        result = result.add_ref(&constant);
    }

    assert_eq!(result.data(), 100.0);

    result.backward();

    // Gradient should be 1 (derivative of sum)
    assert_eq!(x.grad(), 1.0);
}

#[test]
fn test_gradients_accumulate_correctly() {
    // Test: f(x, y) = x*y + x*y (same computation twice)
    let x = value_with_label(3.0, "x");
    let y = value_with_label(4.0, "y");

    let xy1 = x.mul_ref(&y);
    let xy2 = x.mul_ref(&y);
    let result = xy1.add_ref(&xy2);

    assert_eq!(result.data(), 24.0);

    result.backward();

    // z = 2*x*y
    // dz/dx = 2*y = 2*4 = 8
    // dz/dy = 2*x = 2*3 = 6
    assert_eq!(x.grad(), 8.0);
    assert_eq!(y.grad(), 6.0);
}

#[test]
fn test_previous_references() {
    let x = value_with_label(2.0, "x");
    let y = value_with_label(3.0, "y");

    let result = x.add_ref(&y);
    let previous = result.previous();

    assert_eq!(previous.len(), 2);
    assert_eq!(previous[0].data(), 2.0);
    assert_eq!(previous[1].data(), 3.0);
}
