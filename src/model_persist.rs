use anyhow::{bail, Context};
use chrono::prelude::*;
use burn::config::Config;
use burn::module::Module;
use burn::record::Recorder;
use burn::{record::CompactRecorder, tensor::backend::Backend};
use shared_types::paths::artifacts_dir;

use crate::model::{TheModel, TheModelConfig};
use crate::Model;

const CONFIG_MODEL: &str = "config-model.json";

pub fn save_model_config(config: &TheModelConfig) -> anyhow::Result<()> {
    save_config("model", CONFIG_MODEL, config)
}

pub fn load_model_config() -> anyhow::Result<TheModelConfig> {
    load_config("model", CONFIG_MODEL)
}

// pub fn save_model<B:Backend>(model: &TheModel<B>) -> anyhow::Result<()> {
pub fn save_model<B:Backend, M:Model + Module<B>>(model: &M) -> anyhow::Result<()> {
    println!("Saving model at {}...", Local::now());
    // TODO: save only occasionally? and coordinate with commiting offsets
    let path = artifacts_dir()?.join("model");
    // TODO: why have to clone?
    model.clone().save_file(&path, &CompactRecorder::new()).with_context(|| "Error saving model")?;
    println!("   model saved to: {:?}", &path);
    Ok(())
}

pub fn load_model<B:Backend>(device: &B::Device) -> anyhow::Result<TheModel<B>> {
    let base_path = artifacts_dir()?;
    let model_path = base_path.join("model.mpk");
    if model_path.try_exists()? {
        let config_model = load_model_config()?;
        print!("Loading model {:?} ...", &model_path);
        let record = CompactRecorder::new().load(model_path, device).with_context(|| "Error loading model")?;
        let model = config_model.init::<B>(device).load_record(record);
        println!(" model loaded.");
        Ok(model)
    } else {
        bail!("Could not find model path: {:?}", model_path);
    }
}


pub fn save_config<T: Config>(name: &str, filename: &str, config: &T) -> anyhow::Result<()> {
    let config_path = artifacts_dir()?.join(filename);
    // TODO: manage changes to the config
    config.save(&config_path)?;
    println!("{} config saved: {:?}", name, config_path);
    Ok(())
}

pub fn load_config<T: Config>(name: &str, filename: &str) -> anyhow::Result<T> {
    let config_path = artifacts_dir()?.join(filename);
    if config_path.try_exists()? {
        // TODO: manage changes to the config
        let config = T::load(&config_path)?;
        println!("{} config loaded: {:?}", name, config_path);
        Ok(config)
    } else {
        bail!("Load {} config path not found: {:?}", name, config_path);
    }
}
