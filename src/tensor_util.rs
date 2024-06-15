use burn::prelude::*;

// if let Some(x) = check_finite(&out) {
//     println!("Found suspicious value in output {:?}", out.clone().into_data().value);
//     panic!("Found suspicious value in output: {}", x);
// }
// out
pub fn check_finite<B: Backend, const D: usize>(tensor: &Tensor<B,D>) -> Option<f32> {
    for x in tensor.clone().into_data().value {
        let y: f32 = x.elem();
        if !y.is_finite() || !(-1000.0..=1000.0).contains(&y) {
            println!("Found suspicious value in tensor: {}", y);
            return Some(y);
        }
    }
    None
}
