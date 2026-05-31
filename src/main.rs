mod tensor;

use tensor::Tensor;

fn main() {
    let t = Tensor {
        data: vec![1.0, 2.0, 3.0],
        shape: vec![3],
    };

    println!("New tensor {:?}", t.shape);
}
