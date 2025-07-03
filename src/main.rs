use micrograd_rs::Value;

fn main() {
    let x = Value::from(0.5).set_label("x");
    let w = Value::from(3.14).set_label("w");
    let b = Value::from(-2.0).set_label("b");

    let y = (&(&x * &w) + &b).set_label("y");

    println!("{} {}", x.get_data(), x.get_gradient());
    println!("{} {}", w.get_data(), w.get_gradient());
    println!("{} {}", b.get_data(), b.get_gradient());
    println!("{} {}", y.get_data(), y.get_gradient());

    y.backprop();

    println!("{} {}", x.get_data(), x.get_gradient());
    println!("{} {}", w.get_data(), w.get_gradient());
    println!("{} {}", b.get_data(), b.get_gradient());
    println!("{} {}", y.get_data(), y.get_gradient());
}
