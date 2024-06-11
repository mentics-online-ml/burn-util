pub mod model;
pub mod model_persist;
pub mod output;

use burn::prelude::*;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use shared_types::ModelFloat;

pub trait Model {
    type B: Backend;
    type Input: Clone;
    type Output: Clone + IsTensor<Self::B,2>;

    fn forward(&self, input: Self::Input) -> Self::Output;
}

// Tensor<Self::B,3>
// Tensor<Self::B,2>

pub type TheBackend = Wgpu<AutoGraphicsApi, f32, i32>;
pub type TheAutodiffBackend = Autodiff<TheBackend>;

pub trait IsTensor<B: Backend, const D: usize> {
    fn noop(self) -> Tensor<B,D>;
}

impl<B: Backend, const D: usize> IsTensor<B,D> for Tensor<B,D> {
    fn noop(self) -> Tensor<B,D> {
        self
    }
}

pub fn tensor1_to_vec<B:Backend>(tensor: Tensor<B,1>) -> Vec<ModelFloat> {
    tensor.into_data().convert::<f32>().value
}

