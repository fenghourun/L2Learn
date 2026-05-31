use l2learn::tensor::Tensor;

#[test]
fn matmul_2x2() {
    let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);

    let b = Tensor::new(vec![5., 6., 7., 8.], vec![2, 2]);

    let c = a.matmul(&b);

    assert_eq!(c.data, vec![19., 22., 43., 50.]);
}
