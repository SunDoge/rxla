use rxla_core::{Client, Tracer};

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_empty_one_sided_sum_ignores_nonfinite_other_operand() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[0]).unwrap();
    let y = g.input(&[3]).unwrap();
    let output = x.einsum("i,j->j", &y).unwrap();
    let reverse = y.einsum("j,i->j", &x).unwrap();
    let grads = output.sum(&[0], false).unwrap().grad(&[x, y]).unwrap();
    let exe = g
        .compile_many(
            &client,
            &[output, reverse, grads[0].clone(), grads[1].clone()],
        )
        .unwrap();
    let xb = client.buffer::<f32>(&[0], &[]).unwrap();
    let yb = client
        .buffer(&[3], &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY])
        .unwrap();
    let out = exe.execute(&[&xb, &yb]).unwrap();
    assert_eq!(out[0].to_vec::<f32>().unwrap(), [0.; 3]);
    assert_eq!(out[1].to_vec::<f32>().unwrap(), [0.; 3]);
    assert!(out[2].to_vec::<f32>().unwrap().is_empty());
    assert_eq!(out[3].to_vec::<f32>().unwrap(), [0.; 3]);
}

#[test]
fn rejects_invalid_equations_and_dimensions() {
    let g = Tracer::default();
    let x = g.input(&[2, 3]).unwrap();
    let y = g.input(&[3, 2]).unwrap();
    for equation in [
        "ij,jk",
        "ij,jk->ii",
        "ij,jk->z",
        "ij,jk,kl->il",
        "...i,ij->...j",
        "i,jk->jk",
        "ii,jk->jk",
        "ij,ij->ij",
        "ij,jk->é",
    ] {
        assert!(x.einsum(equation, &y).is_err(), "{equation}");
    }
    assert!(
        x.einsum("ij,jk->ik", &Tracer::default().input(&[3, 2]).unwrap())
            .is_err()
    );
    assert_eq!(x.einsum(" iJ , Jk -> ki ", &y).unwrap().shape(), [2, 2]);
}

// Dense F64 label enumeration, independent of transpose/diagonal/matmul lowering.
fn reference(
    left: &str,
    right: &str,
    output: &str,
    ls: &[i64],
    rs: &[i64],
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    let mut dims = [0usize; 128];
    let mut labels = Vec::new();
    for (name, shape) in [(left, ls), (right, rs)] {
        for (label, &size) in name.bytes().zip(shape) {
            if !labels.contains(&label) {
                labels.push(label);
            }
            dims[label as usize] = size as usize;
        }
    }
    let size: usize = output.bytes().map(|l| dims[l as usize]).product();
    let mut expected = vec![0.0f64; size];
    let count: usize = labels.iter().map(|&l| dims[l as usize]).product();
    for linear in 0..count {
        let mut position = [0usize; 128];
        let mut n = linear;
        for &label in labels.iter().rev() {
            position[label as usize] = n % dims[label as usize];
            n /= dims[label as usize];
        }
        let index = |name: &str| {
            name.bytes()
                .fold(0, |i, l| i * dims[l as usize] + position[l as usize])
        };
        expected[index(output)] += f64::from(x[index(left)]) * f64::from(y[index(right)]);
    }
    expected.into_iter().map(|v| v as f32).collect()
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_einsum_matches_dense_label_enumeration() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let cases: &[(&str, &[i64], &[i64])] = &[
        ("bmk,kbn->nbm", &[2, 3, 4], &[4, 2, 5]),
        ("ii,i->i", &[3, 3], &[3]),
        ("iii,->i", &[3, 3, 3], &[]),
        ("ij,k->i", &[2, 3], &[4]),
        ("ij,jk->", &[2, 3], &[3, 4]),
        ("i,j->ji", &[3], &[2]),
        (",->", &[], &[]),
        ("ij,jk->ik", &[2, 0], &[0, 3]),
        ("bi,bi->b", &[0, 3], &[0, 3]),
        ("ii,jj->", &[0, 0], &[2, 2]),
    ];
    for &(equation, ls, rs) in cases {
        let g = Tracer::default();
        let lhs = g.input(ls).unwrap();
        let rhs = g.input(rs).unwrap();
        let result = lhs.einsum(equation, &rhs).unwrap();
        let exe = g.compile(&client, &result).unwrap();
        let x: Vec<f32> = (0..ls.iter().product::<i64>())
            .map(|i| (i as f32 % 13. - 6.) * 0.125)
            .collect();
        let y: Vec<f32> = (0..rs.iter().product::<i64>())
            .map(|i| (i as f32 % 11. - 5.) * 0.25)
            .collect();
        let xb = client.buffer(ls, &x).unwrap();
        let yb = client.buffer(rs, &y).unwrap();
        let actual = exe.execute(&[&xb, &yb]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap();
        let (input, output) = equation.split_once("->").unwrap();
        let (left, right) = input.split_once(',').unwrap();
        assert_eq!(
            actual,
            reference(left, right, output, ls, rs, &x, &y),
            "{equation}"
        );
    }
}

#[test]
#[ignore = "requires trusted PJRT_PLUGIN_PATH"]
fn real_einsum_diagonal_contraction_higher_derivatives() {
    let client = unsafe { Client::load(std::env::var("PJRT_PLUGIN_PATH").unwrap()) }.unwrap();
    let g = Tracer::default();
    let x = g.input(&[2, 2]).unwrap();
    let w = g.input(&[2]).unwrap();
    let y = x.einsum("ii,i->", &w).unwrap();
    let grads = y.mul(&y).unwrap().grad(&[x.clone(), w]).unwrap();
    let second = grads[0]
        .sum(&[0, 1], false)
        .unwrap()
        .grad(&[x])
        .unwrap()
        .remove(0);
    let exe = g
        .compile_many(&client, &[y, grads[0].clone(), grads[1].clone(), second])
        .unwrap();
    let xb = client.buffer(&[2, 2], &[1., 9., 8., 2.]).unwrap();
    let wb = client.buffer(&[2], &[3., 4.]).unwrap();
    let out = exe.execute(&[&xb, &wb]).unwrap();
    for (actual, expected) in out.iter().zip([
        vec![11.],
        vec![66., 0., 0., 88.],
        vec![22., 44.],
        vec![42., 0., 0., 56.],
    ]) {
        assert_eq!(actual.to_vec::<f32>().unwrap(), expected);
    }
}
