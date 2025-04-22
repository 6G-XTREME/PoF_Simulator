use pyo3::prelude::*;

#[pyfunction]
fn say_hello() -> PyResult<String> {
    Ok("Hello from Rust!".to_string())
}

#[pyfunction]
fn say_hello_stdin() -> PyResult<()> {
    use std::io::{self, Write};
    let mut input = String::new();
    print!("Enter your name: ");
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut input).unwrap();
    let name = input.trim();
    println!("Hello, {}!", name);
    Ok(())
}



#[pymodule]
fn mi_modulo(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(say_hello, m)?)?;
    m.add_function(wrap_pyfunction!(say_hello_stdin, m)?)?;
    Ok(())
}