use std::ops::{Add, Div, Mul, Sub};

pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_size: usize = shape.iter().product();

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

    pub fn has_nan(&self) -> bool {
        self.data.iter().any(|x| x.is_nan())
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);

        let n = self.shape[0];
        let k = self.shape[1];
        let k2 = other.shape[0];
        let m = other.shape[1];

        assert_eq!(
            k, k2,
            "Cannot multiply {:?} by {:?}",
            self.shape, other.shape
        );

        let mut data = vec![0.0; m * n];

        for i in 0..n {
            for j in 0..m {
                // Construct the (i, j) th element of the product matrix
                let mut sum = 0.0;
                for t in 0..k {
                    sum += self.get_element_at(i, t) * other.get_element_at(t, j);
                }
                data[i * n + j] = sum;
            }
        }

        Tensor::new(data, vec![n, m])
    }

    pub fn transpose(&self) -> Tensor {
        let n = self.shape[0];
        let m = self.shape[1];
        let mut transpose = Tensor::new(vec![0.0; n * m], vec![m, n]);

        for i in 0..n {
            for j in 0..m {
                let element = self.get_element_at(i, j);
                transpose.set_element_at(j, i, element);
            }
        }

        transpose
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        if self.data.len() == 0 {
            f32::NAN
        } else {
            let sum: f32 = self.data.iter().sum();
            sum / self.data.len() as f32
        }
    }

    pub fn var(&self) -> f32 {
        let mean = self.mean();
        let numerator: f32 = self.data.iter().map(|x| (x - mean).powi(2)).sum();

        numerator / self.data.len() as f32
    }

    /// For a 2D matrix return the element corresponding to the (i, j)'th element
    fn get_element_at(&self, i: usize, j: usize) -> f32 {
        let n = self.shape[0];
        let m = self.shape[1];

        assert!(i <= n, "Invalid index i {:?} for shape {:?}", i, self.shape);
        assert!(j <= m, "Invalid index j {:?} for shape {:?}", j, self.shape);

        self.data[i * m + j]
    }

    /// For a 2D matrix set the element corresponding to the (i, j)'th element
    fn set_element_at(&mut self, i: usize, j: usize, value: f32) {
        let n = self.shape[0];
        let m = self.shape[1];

        assert!(i <= n, "Invalid index i {:?} for shape {:?}", i, self.shape);
        assert!(j <= m, "Invalid index j {:?} for shape {:?}", j, self.shape);

        self.data[i * m + j] = value;
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "Cannot add {:?} with {:?}",
            self.shape, other.shape
        );

        Tensor::new(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
            self.shape,
        )
    }
}

impl Add<f32> for Tensor {
    type Output = Tensor;
    fn add(self, other: f32) -> Tensor {
        Tensor::new(
            self.data.iter().map(|x| x + other).collect(),
            self.shape.clone(),
        )
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, other: Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "Cannot sub {:?} with {:?}",
            self.shape, other.shape
        );

        Tensor::new(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
            self.shape,
        )
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;
    fn sub(self, other: f32) -> Tensor {
        Tensor::new(
            self.data.iter().map(|x| x - other).collect(),
            self.shape.clone(),
        )
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "Cannot elementwise multiply {:?} with {:?}",
            self.shape, other.shape
        );

        Tensor::new(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .collect(),
            self.shape,
        )
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;
    fn mul(self, other: f32) -> Tensor {
        Tensor::new(
            self.data.iter().map(|x| x * other).collect(),
            self.shape.clone(),
        )
    }
}

impl Div for Tensor {
    type Output = Tensor;
    fn div(self, other: Tensor) -> Tensor {
        assert_eq!(
            self.shape, other.shape,
            "Cannot elementwise divide {:?} with {:?}",
            self.shape, other.shape
        );

        Tensor::new(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a / b)
                .collect(),
            self.shape,
        )
    }
}

impl Div<f32> for Tensor {
    type Output = Tensor;
    fn div(self, other: f32) -> Tensor {
        Tensor::new(
            self.data.iter().map(|x| x / other).collect(),
            self.shape.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_creation() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(t.shape, vec![3]);
    }
    #[test]
    fn has_nan() {
        let t = Tensor::new(vec![1.0, 2.0, f32::NAN], vec![3]);
        assert!(t.has_nan());
    }
    #[test]
    fn matmul_1x1() {
        let a = Tensor::new(vec![3.], vec![1, 1]);
        let b = Tensor::new(vec![7.], vec![1, 1]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![21.]);
    }
    #[test]
    fn matmul_2x2() {
        let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let b = Tensor::new(vec![5., 6., 7., 8.], vec![2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![19., 22., 43., 50.]);
    }
    #[test]
    fn matmul_3x3() {
        let a = Tensor::new(vec![1., 2., 3., 4., 5., 6., 7., 8., 9.], vec![3, 3]);
        let b = Tensor::new(vec![9., 8., 7., 6., 5., 4., 3., 2., 1.], vec![3, 3]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![30., 24., 18., 84., 69., 54., 138., 114., 90.,]);
    }

    #[test]
    fn matmul_rectangular() {
        let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = Tensor::new(vec![7., 8., 9., 10., 11., 12.], vec![3, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![58., 64., 139., 154.,]);
    }

    #[test]
    #[should_panic(expected = "Cannot multiply [2, 3] by [2, 2]")]
    fn matmul_invalid_shapes() {
        let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = Tensor::new(vec![7., 8., 9., 10.], vec![2, 2]);
        a.matmul(&b);
    }

    #[test]
    fn transpose_1x1() {
        let a = Tensor::new(vec![5.], vec![1, 1]);
        let t = a.transpose();
        assert_eq!(t.data, vec![5.]);
    }

    #[test]
    fn transpose_2x2() {
        let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let t = a.transpose();
        assert_eq!(t.data, vec![1., 3., 2., 4.]);
    }

    #[test]
    fn transpose_2x3() {
        let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let t = a.transpose();
        assert_eq!(t.data, vec![1., 4., 2., 5., 3., 6.]);
        assert_eq!(t.shape, vec![3, 2]);
    }

    #[test]
    fn add_2x2() {
        let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let b = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let c = a + b;
        assert_eq!(c.data, vec![2., 4., 6., 8.]);
        assert_eq!(c.shape, vec![2, 2]);
    }

    #[test]
    #[should_panic(expected = "Cannot add [2, 3] with [2, 2]")]
    fn add_invalid_shapes() {
        let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let _ = a + b;
    }

    #[test]
    fn add_scalar() {
        let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let b = 10.;
        let c = a + b;
        assert_eq!(c.data, vec![11., 12., 13., 14.]);
        assert_eq!(c.shape, vec![2, 2]);
    }

    #[test]
    fn sub_2x2() {
        let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let b = Tensor::new(vec![1., 1., 3., 5.], vec![2, 2]);
        let c = a - b;
        assert_eq!(c.data, vec![0., 1., 0., -1.]);
        assert_eq!(c.shape, vec![2, 2]);
    }

    #[test]
    #[should_panic(expected = "Cannot sub [2, 3] with [2, 2]")]
    fn sub_invalid_shapes() {
        let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let _ = a - b;
    }

    #[test]
    fn sub_scalar() {
        let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let b = 10.;
        let c = a - b;
        assert_eq!(c.data, vec![-9., -8., -7., -6.]);
        assert_eq!(c.shape, vec![2, 2]);
    }

    #[test]
    fn mul_2x2() {
        let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let b = Tensor::new(vec![1., 1., 3., -5.], vec![2, 2]);
        let c = a * b;
        assert_eq!(c.data, vec![1., 2., 9., -20.]);
        assert_eq!(c.shape, vec![2, 2]);
    }

    #[test]
    #[should_panic(expected = "Cannot elementwise multiply [2, 3] with [2, 2]")]
    fn mul_invalid_shapes() {
        let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let _ = a * b;
    }

    #[test]
    fn mul_scalar() {
        let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let b = 10.;
        let c = a * b;
        assert_eq!(c.data, vec![10., 20., 30., 40.]);
        assert_eq!(c.shape, vec![2, 2]);
    }

    #[test]
    fn div_2x2() {
        let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let b = Tensor::new(vec![1., 1., 3., -5.], vec![2, 2]);
        let c = a / b;
        assert_eq!(c.data, vec![1., 2., 1., -0.8]);
        assert_eq!(c.shape, vec![2, 2]);
    }

    #[test]
    #[should_panic(expected = "Cannot elementwise divide [2, 3] with [2, 2]")]
    fn div_invalid_shapes() {
        let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let _ = a / b;
    }

    #[test]
    fn div_scalar() {
        let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let b = 10.;
        let c = a / b;
        assert_eq!(c.data, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(c.shape, vec![2, 2]);
    }

    #[test]
    fn mean_2x2() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let mean = t.mean();
        assert_eq!(mean, 2.5);
    }

    #[test]
    fn mean_contains_nan_propagates() {
        let t = Tensor::new(vec![1.0, f32::NAN, 3.0, 4.0], vec![2, 2]);
        let mean = t.mean();
        assert!(mean.is_nan());
        assert!(t.has_nan());
    }

    #[test]
    fn var_2x2() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let var = t.var();
        assert_eq!(var, 1.25);
    }

    #[test]
    fn var_contains_nan_propagates() {
        let t = Tensor::new(vec![1.0, 2.0, f32::NAN, 4.0], vec![2, 2]);

        let var = t.var();

        assert!(var.is_nan());
        assert!(t.has_nan());
    }
}
