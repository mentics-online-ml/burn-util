use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

// pub trait HasForward<B: Backend, const D: usize, R> {
//     fn forward(&self, input: Tensor<B, D>) -> R
//     where [(); D-1]:; //Tensor<B, D>;
// }

// impl<B: Backend> HasForward<B> for Linear<B> {
//     fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
//         self.forward(input)
//     }
// }

#[derive(Module, Debug)]
pub struct OneAcrossDim<B: Backend> {
    pub block: Block<B>
}

impl<B: Backend> OneAcrossDim<B> {
    pub fn new(block: Block<B>) -> Self {
        Self { block }
    }

    pub fn forward<const D: usize>(&self, input: Tensor<B, D>, dim: usize) -> Tensor<B, D>
    where [(); D-1]: {
        let tensors_across_dim = input.iter_dim(dim).map(|t| {
            let t2: Tensor<B, {D-1}> = t.squeeze(dim);
            let t3 = self.block.forward(t2);
            t3.unsqueeze_dim::<D>(dim)
        }).collect();
        Tensor::cat(tensors_across_dim, dim)
    }
}

#[derive(Module, Debug)]
pub struct AcrossDim<B: Backend> {
    modules: Vec<Block<B>>
}

impl<B: Backend> AcrossDim<B> {
    pub fn new<F: FnMut(usize) -> Block<B>>(count: usize, f: F) -> Self {
        let v = (0..count).map(f).collect();
        AcrossDim { modules: v }
    }

    pub fn forward<const D: usize>(&self, input: Tensor<B, D>, dim: usize) -> Tensor<B, D>
    where [(); D-1]: {
        let b = input.clone().iter_dim(dim).enumerate().map(|(i, t)| {
            let t2: Tensor<B, {D-1}> = t.squeeze(dim);
            let t3 = self.modules[i].forward(t2);
            t3.unsqueeze_dim::<D>(dim)
        });
        let c = b.collect();
        Tensor::cat(c, dim)
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub modules: Vec<Linear<B>>,
    activation: Mish
}

impl<B: Backend> Block<B> {
    pub fn new<F: FnMut(usize) -> Linear<B>>(count: usize, f: F) -> Self {
        let modules = (0..count).map(f).collect();
        Block { modules, activation: Mish::default() }
    }

    pub fn new_linear(device: &B::Device,
            count: usize, input_width: usize, hidden_width: usize, output_width: usize, bias: bool) -> Self {
        assert!(count >= 2);
        let mut modules = Vec::with_capacity(count);
        modules.push(LinearConfig::new(input_width, hidden_width).with_bias(bias).init(device));
        for _ in 1..(count-1) {
            modules.push(LinearConfig::new(hidden_width, hidden_width).with_bias(bias).init(device));
        }
        modules.push(LinearConfig::new(hidden_width, output_width).with_bias(bias).init(device));
        Block { modules, activation: Mish::default() }
    }

    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.modules.iter().fold(input, |acc, m| self.activation.forward(m.forward(acc)))
        // self.modules.iter().fold(input, |acc, m| m.forward(acc))
    }

    pub fn shape(&self) -> Shape<3> {
        let dims0 = self.modules.first().unwrap().weight.shape().dims;
        let dims_end = self.modules.last().unwrap().weight.shape().dims;
        Shape::new([dims0[0], dims0[1], dims_end[1]])
    }

    pub fn shape_str(&self) -> String {
        let dims = self.shape().dims;
        format!("{:?} -> {:?} -> {:?}", dims[0], dims[1], dims[2])
    }
}

#[derive(Module, Clone, Debug, Default)]
pub struct Mish {}

// impl<B: Backend, const D: usize> HasForward<B, D, Tensor<B, D>> for Mish {
impl Mish {
    /// Applies the mish function on each element of the tensor.
    fn forward<B:Backend,const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        burn::tensor::activation::mish(input)
    }
}
