use burn::{nn::{Linear, LinearConfig}, prelude::*, tensor::activation::mish};
use layers::{AcrossDim, Block, OneAcrossDim};
use nn::{Dropout, DropoutConfig};

use shared_types::{*, chrono_util::CHRONO_FEATURES_SIZE};
use data_info::*;
use crate::*;

pub type ModelInput<B> = (Tensor<B,2>, Tensor<B,3>);
pub type ModelOutput<B> = Tensor<B,2>;

pub const STREAM_EVENT_DIM: usize = 0;
pub const STREAM_FEATURE_DIM: usize = 1;

pub const INPUT_WIDTH: usize = CHRONO_FEATURES_SIZE + SERIES1_SIZE * (TIME_EMBEDDING_SIZE + SERIES1_FEATURES_SIZE);
pub const HIDDEN_MULT: f32 = 1.8;
pub const OUTPUT_MULT: f32 = 0.7;
// pub const BLOCK_DEPTH: usize = 2;

impl Default for NewModelConfig {
    fn default() -> Self {
        let block_depth = 2;
        let stream_size = StreamSize::new(SERIES1_ITEM_SIZE, SERIES1_SIZE);
        Self { stream_size, block_depth, dropout: 0.2 }
    }
}

fn new_block<B: Backend>(device: &B::Device, block_depth: usize, input: usize, hidden: usize, output: usize) -> OneAcrossDim<B> {
    OneAcrossDim::new(Block::new_linear(device, block_depth, input, hidden, output, false))
}

fn new_across_block<B: Backend>(device: &B::Device, module_count: usize, block_depth: usize, input: usize, hidden: usize, output: usize) -> AcrossDim<B> {
    AcrossDim::new(module_count, |_| Block::new_linear(device, block_depth, input, hidden, output, false))
}

#[derive(Module,Clone,Debug,serde::Serialize,serde::Deserialize)]
struct StreamSize {
    item_size: usize,
    series_size: usize,

    // event_dim: usize,
    // feature_dim: usize,

    item_hidden_size: usize,
    item_output_size: usize,
    series_hidden_size: usize,
    series_output_size: usize,
}

/// event/feature_dim are hardcoded because all streams
impl StreamSize {
    fn new(item_size: usize, series_size: usize) -> Self {
        Self {
            // event_dim: STREAM_EVENT_DIM,
            // feature_dim: STREAM_FEATURE_DIM,
            item_size, series_size,
            item_hidden_size: round_mult(HIDDEN_MULT, item_size),
            item_output_size: round_mult(OUTPUT_MULT, item_size),
            series_hidden_size: round_mult(HIDDEN_MULT, series_size),
            series_output_size: round_mult(OUTPUT_MULT, series_size),
        }
    }
}

#[derive(Module, Debug)]
struct StreamModel<B:Backend> {
    size: StreamSize,
    item: OneAcrossDim<B>,
    feature: AcrossDim<B>,
}

impl<B:Backend> StreamModel<B> {
    fn new(device: &B::Device, block_depth: usize, stream_size: StreamSize) -> Self {
        let item = new_block(device, block_depth, stream_size.item_size, stream_size.item_hidden_size, stream_size.item_output_size);
        let feature = new_across_block(device, stream_size.item_size, block_depth, stream_size.series_size, stream_size.series_hidden_size, stream_size.series_output_size);
        Self { size: stream_size, item, feature }
    }
}

// impl<B:Backend,const D: usize> HasForward<B,D,Tensor<B, 1>> for StreamModel<B> {
impl<B:Backend> StreamModel<B> {
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> (Tensor<B, D>, Tensor<B, D>)
    where [(); D-1]: {
        let new_item = self.item.forward(input.clone(), 1 + STREAM_EVENT_DIM);
        let new_feature = self.feature.forward(input, 1 + STREAM_FEATURE_DIM);
        // Tensor::cat(vec![new_item.flatten(1, D), new_feature.flatten(1, D)], 0)
        (new_item, new_feature)
    }
}

pub type TheModelConfig = NewModelConfig;
pub type TheModel<B> = NewModel<B>;

#[derive(Config)]
pub struct NewModelConfig {
    stream_size: StreamSize,
    block_depth: usize,
    dropout: f64,
}

impl NewModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NewModel<B> {
        let dropout = DropoutConfig::new(self.dropout).init();

        let flat_len_sum = self.stream_size.series_size * self.stream_size.item_output_size + self.stream_size.series_output_size * self.stream_size.item_size;
        let output_input_size = CHRONO_FEATURES_SIZE + flat_len_sum;

        let output = Block::new_linear(device, self.block_depth, output_input_size, output_input_size, MODEL_OUTPUT_WIDTH, false);
        // LinearConfig::new(output_input_size, MODEL_OUTPUT_WIDTH).with_bias(false).init(device);
        let stream_model = StreamModel::new(device, self.block_depth, self.stream_size.clone());

        NewModel { stream: stream_model, dropout, output }
    }
}

#[derive(Module,Debug)]
pub struct NewModel<B:Backend> {
    stream: StreamModel<B>,
    dropout: Dropout,
    output: Block<B>,
}

impl<B:Backend> Model for NewModel<B> {
    type B = B;
    type Input = ModelInput<B>;
    type Output = ModelOutput<B>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        // println!("item weight dims: {:?}", self.stream.item.block.modules.last().unwrap().weight.dims());
        assert!(self.stream.item.block.modules[0].bias.is_none());

        let batch_size = input.0.dims()[0];
        let input_stream1 = input.1;

        assert!(input_stream1.dims() == [batch_size, self.stream.size.series_size, self.stream.size.item_size]);
        let (item_tensor, feature_tensor) = self.stream.forward(input_stream1);
        // println!("stream_size: {:?}", self.stream.size);
        // println!("batch_size: {}, dims: item: {:?}, feature: {:?}", batch_size, item_tensor.dims(), feature_tensor.dims());
        assert!(item_tensor.dims() == [batch_size, self.stream.size.series_size, self.stream.size.item_output_size]);
        assert!(feature_tensor.dims() == [batch_size, self.stream.size.series_output_size, self.stream.size.item_size]);

        let item_flat = item_tensor.reshape([0,-1]);
        let feature_flat = feature_tensor.reshape([0,-1]);
        let flat_len_sum = self.stream.size.series_size * self.stream.size.item_output_size + self.stream.size.series_output_size * self.stream.size.item_size;
        let input0_len = input.0.dims()[1];

        let combined = Tensor::cat(vec![input.0, item_flat, feature_flat], 1);
        assert!(combined.dims() == [batch_size, input0_len + flat_len_sum]);

        activation_normalization(self.output.forward(combined))
    }
}

#[derive(Config, Debug)]
pub struct OldModelConfig {
    input_width: usize,
    hidden_width: usize,
    output_width: usize,

    #[config(default = "2.0")]
    features_width_mult: f32,
    #[config(default = "2.0")]
    events_width_mult: f32,
    #[config(default = "2.0")]
    hidden_width_mult: f32,
    #[config(default = "4")]
    block_depth: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

impl OldModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> OldModel<B> {
        // let output = LinearConfig::new(self.num_features * self.SERIES1_LENGTH, self.num_classes).init(device);
        let hidden_width = 8192;
        let bias = false;
        let input = LinearConfig::new(self.input_width, hidden_width / 2).with_bias(bias).init(device);
        let layer1 = LinearConfig::new(hidden_width / 2, hidden_width).with_bias(bias).init(device);
        let layer3 = LinearConfig::new(hidden_width, 2 * hidden_width).with_bias(bias).init(device);
        let layer4 = LinearConfig::new(2 * hidden_width, hidden_width).with_bias(bias).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();
        let layer5 = LinearConfig::new(hidden_width, 2 * hidden_width).with_bias(bias).init(device);
        let layer6 = LinearConfig::new(2 * hidden_width, hidden_width).with_bias(bias).init(device);
        let layer2 = LinearConfig::new(hidden_width, hidden_width / 2).with_bias(bias).init(device);
        let output = LinearConfig::new(hidden_width / 2, self.output_width).with_bias(bias).init(device);
        OldModel { input, layer1, dropout, layer2, output, layer3, layer4, layer5, layer6 }
    }
}

impl OldModelConfig {
    pub fn new_config() -> OldModelConfig {
        OldModelConfig::new(INPUT_WIDTH, 4 * INPUT_WIDTH, MODEL_OUTPUT_WIDTH)
    }
}

#[derive(Module, Debug)]
pub struct OldModel<B: Backend> {
    input: Linear<B>,
    layer1: Linear<B>,
    dropout: Dropout,
    layer2: Linear<B>,
    output: Linear<B>,
    layer3: Linear<B>,
    layer4: Linear<B>,
    layer5: Linear<B>,
    layer6: Linear<B>,
}

/// # Shapes
///   - Input [batch_size, SERIES1_LENGTH, NUM_FEATURES]
///   - Output [batch_size, MODEL_OUTPUT_WIDTH]
impl<B:Backend> Model for OldModel<B> {
    type B = B;
    type Input = ModelInput<B>;
    type Output = ModelOutput<B>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        // let a = self.output.forward(input.flatten(1, 2));
        // self.act.forward(a)

        let input_flattened = Tensor::cat(vec![input.0, input.1.flatten(1, 2)], 1);
        let mut x = self.input.forward(input_flattened);
        x = activation_normalization(x);
        x = self.layer1.forward(x);
        x = activation_normalization(x);
        let layer1_output = x.clone();

        x = self.layer3.forward(x);
        x = activation_normalization(x);
        x = self.layer4.forward(x);
        x = activation_normalization(x);
        x = x.add(layer1_output);

        x = self.dropout.forward(x);

        x = self.layer5.forward(x);
        x = activation_normalization(x);
        x = self.layer6.forward(x);
        x = activation_normalization(x);
        x = self.layer2.forward(x);
        x = activation_normalization(x);
        x = self.output.forward(x);
        x = activation_normalization(x);
        x

        // burn::tensor::activation::softmax(c, 0)
        // burn::tensor::activation::mish(c).clamp(0f32, 1f32)
        // burn::tensor::activation::mish(c)
        // self.act.forward(c)
    }
}

fn activation_normalization<B:Backend>(tensor: Tensor<B,2>) -> Tensor<B,2> {
    mish(tensor).clamp(0.00001, 0.99999)
}

// fn run_in_order<B:Backend,const D: usize,T:Default>(funcs: Vec<impl FnOnce(T) -> T>) -> T {
//     let mut x = Default::default();
//     for f in funcs {
//         x = f(x);
//     }
//     x
// }

pub fn inputs_to_device<B: Backend>(inputs: &Vec<InputRaw>, device: &B::Device) -> ModelInput<B> {
    let batch_size = inputs.len();
    let mut chronos_flat = Vec::new();
    let mut serieses_flat = Vec::new();
    for (chrono, series) in inputs {
        chronos_flat.extend(chrono);
        serieses_flat.append(&mut series.concat());
    }

    // let (ref chronos_raw, ref serieses_raw): (Vec<_>, Vec<&SeriesItem>) = inputs.iter().map(distribute_ref).unzip();
    // let chronos_data = Data::new(chronos_raw.concat(), Shape::new([batch_size, CHRONO_FEATURES_SIZE]));
    // let serieses_data = Data::new(serieses_raw.concat::<SeriesItem>().concat(), Shape::new([batch_size, SERIES1_SIZE, SERIES1_ITEM_SIZE]));
    let chronos_data = Data::new(chronos_flat, Shape::new([batch_size, CHRONO_FEATURES_SIZE]));
    let serieses_data = Data::new(serieses_flat, Shape::new([batch_size, SERIES1_SIZE, SERIES1_ITEM_SIZE]));
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

fn round_mult(a: f32, b: usize) -> usize {
    (a * b as f32).ceil() as usize
}
