pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_size = shape.iter().product();

        assert_eq!(
            data.len(),
            expected_size,
            "Data length ({}) does not match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_size
        );

        Self { data, shape }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_creation() {
        let t = Tensor {
            data: vec![1.0, 2.0, 3.0],
            shape: vec![3],
        };

        assert_eq!(t.shape, vec![3]);
    }
}
