use burn::{nn::{Linear, LinearConfig, Relu}, prelude::*, tensor::activation::mish};
use nn::{Dropout, DropoutConfig};

use shared_types::{*, chrono_util::CHRONO_FEATURES_SIZE};
use data_info::*;
use crate::*;


// pub type ModelInput1<B> = (Tensor<B,1>, Tensor<B,2>);
// pub type ModelOutput1<B> = Tensor<B,1>;

pub type ModelInput<B> = (Tensor<B,2>, Tensor<B,3>);
pub type ModelOutput<B> = Tensor<B,2>;

pub fn inputs_to_device<B: Backend>(inputs: Vec<InputRaw>, device: &B::Device) -> ModelInput<B> {
    let batch_size = inputs.len();
    let (chronos_raw, serieses_raw): (Vec<_>, Vec<_>) = inputs.into_iter().unzip();
    let chronos_data = Data::new(chronos_raw.concat(), Shape::new([batch_size, CHRONO_FEATURES_SIZE]));
    let serieses_data = Data::new(serieses_raw.concat().concat(), Shape::new([batch_size, SERIES1_SIZE, SERIES1_ITEM_SIZE]));
    let chronos = Tensor::from_floats(chronos_data, device);
    let serieses = Tensor::from_floats(serieses_data, device);
    // println!("batch_size {}", batch_size);
    // println!("serieses sizes {}, {}, {} -> {}", batch_size, SERIES1_SIZE, SERIES1_ITEM_SIZE, batch_size*SERIES1_SIZE*SERIES1_ITEM_SIZE);
    // println!("chronos_raw.len {}", chronos_raw.len());
    // println!("serieses_raw.len {}", serieses_raw.len());
    // println!("chronos_data.len {:?}", chronos_data.shape);
    // println!("serieses_data.len {:?}", serieses_data.shape);
    // println!("chronos shape: {:?}", chronos.shape());
    // println!("serieses shape: {:?}", serieses.shape());
    (chronos, serieses)
}

pub fn outputs_to_device<B: Backend>(outputs_raw: Vec<LabelType>, device: &B::Device) -> ModelOutput<B> {
    let batch_size = outputs_raw.len();
    let outputs_data = Data::new(outputs_raw.concat(), Shape::new([batch_size, MODEL_OUTPUT_WIDTH]));
    Tensor::from_floats(outputs_data, device)
}

// pub struct ModelInputRawStruct<const N: usize>(ModelInputRaw<N>);
// pub struct ModelInputStruct<B:Backend>(ModelInput<B>);

// impl<B:Backend, const N: usize> From<ModelInputRawStruct<N>> for ModelInputStruct<B> {
//     fn from((chrono, series): ModelInputRawStruct<N>) -> Self {
//         ModelInputStruct((Tensor::from_floats(chrono), Tensor::from_floats(series)));
//     }
// }


pub const INPUT_WIDTH: usize = CHRONO_FEATURES_SIZE + SERIES1_SIZE * (TIME_ENCODING_SIZE + SERIES1_FEATURES_SIZE);
pub const DROPOUT_FACTOR: f64 = 0.2;

#[derive(Config, Debug)]
pub struct TheModelConfig {
    input_width: usize,
    hidden_width: usize,
    output_width: usize,
    dropout_factor: f64,
}

impl TheModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TheModel<B> {
        // let output = LinearConfig::new(self.num_features * self.SERIES1_LENGTH, self.num_classes).init(device);
        let hidden_width = 8192;
        let input = LinearConfig::new(self.input_width, hidden_width).init(device);
        let layer1 = LinearConfig::new(hidden_width, hidden_width).init(device);
        let dropout = DropoutConfig::new(self.dropout_factor).init();
        let layer2 = LinearConfig::new(hidden_width, hidden_width).init(device);
        let output = LinearConfig::new(hidden_width, self.output_width).init(device);
        TheModel { input, layer1, dropout, layer2, output, act: Relu::new() }
    }
}

impl TheModelConfig {
    pub fn new_config() -> TheModelConfig {
        TheModelConfig::new(INPUT_WIDTH, 4 * INPUT_WIDTH, MODEL_OUTPUT_WIDTH, DROPOUT_FACTOR)
    }
}

#[derive(Module, Debug)]
pub struct TheModel<B: Backend> {
    input: Linear<B>,
    layer1: Linear<B>,
    dropout: Dropout,
    layer2: Linear<B>,
    output: Linear<B>,
    act: Relu
}

/// # Shapes
///   - Input [batch_size, SERIES1_LENGTH, NUM_FEATURES]
///   - Output [batch_size, MODEL_OUTPUT_WIDTH]

impl<B:Backend> Model for TheModel<B> {
    type B = B;
    type Input = ModelInput<B>;
    type Output = ModelOutput<B>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        // let a = self.output.forward(input.flatten(1, 2));
        // self.act.forward(a)

        let input_flattened = Tensor::cat(vec![input.0, input.1.flatten(1, 2)], 1);
        let mut x = self.input.forward(input_flattened);
        x = mish(x);
        x = self.layer1.forward(x);
        x = mish(x);
        x = self.dropout.forward(x);
        x = self.layer2.forward(x);
        x = mish(x);
        x = self.output.forward(x);
        x = mish(x);
        x

        // burn::tensor::activation::softmax(c, 0)
        // burn::tensor::activation::mish(c).clamp(0f32, 1f32)
        // burn::tensor::activation::mish(c)
        // self.act.forward(c)
    }
}

// fn run_in_order<B:Backend,const D: usize,T:Default>(funcs: Vec<impl FnOnce(T) -> T>) -> T {
//     let mut x = Default::default();
//     for f in funcs {
//         x = f(x);
//     }
//     x
// }
