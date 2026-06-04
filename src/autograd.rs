use crate::tensor::Tensor;

enum Op {
    Add,
    Mul,
    MatMul,
    Mean,
    Relu,
}

pub struct Node {
    parents: Vec<Tensor>,
    op: Op,
}
