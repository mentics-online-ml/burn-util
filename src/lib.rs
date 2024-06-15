#![feature(generic_const_exprs)]

pub mod tensor_util;
pub mod layers;
pub mod model;
pub mod model_persist;
pub mod output;

use burn::backend::Autodiff;
use burn::prelude::*;
// use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use shared_types::data_info::{LabelType, MODEL_OUTPUT_WIDTH};
use shared_types::ModelFloat;

pub trait Model {
    type B: Backend;
    type Input: Clone;
    type Output: Clone + IsTensor<Self::B,2>;

    fn forward(&self, input: Self::Input) -> Self::Output;
}

// Tensor<Self::B,3>
// Tensor<Self::B,2>

// pub fn burn_device() -> burn::backend::wgpu::WgpuDevice {
//     burn::backend::wgpu::WgpuDevice::default()
// }
pub fn burn_device() -> burn::backend::candle::CandleDevice {
    burn::backend::candle::CandleDevice::Cuda(0)
}

// pub type TheBackend = Wgpu<AutoGraphicsApi, f32, i32>;
pub type TheBackend = burn::backend::Candle;
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

pub fn tensor2_to_label_vec<B:Backend>(tensor: Tensor<B,2>) -> Vec<LabelType> {
    let data = tensor.into_data();
    assert!(data.shape.dims[1] == MODEL_OUTPUT_WIDTH); // makes unwrap safe
    data.convert::<f32>().value.chunks_exact(MODEL_OUTPUT_WIDTH)
            .map(|chunk| chunk.try_into().unwrap()).collect()
}
