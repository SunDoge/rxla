#[test]
fn layer_attribute_diagnostics() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/model_*.rs");
}
