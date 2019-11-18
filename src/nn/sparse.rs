//! Sparse Layers
use crate::{Device, Tensor, ToDevice};
use std::borrow::Borrow;

/// Configuration option for an embedding layer.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingConfig {
    pub sparse: bool,
    pub scale_grad_by_freq: bool,
    pub ws_init: super::Init,
    pub padding_idx: i64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            sparse: false,
            scale_grad_by_freq: false,
            ws_init: super::Init::Randn {
                mean: 0.,
                stdev: 1.,
            },
            padding_idx: -1,
        }
    }
}

/// An embedding layer.
///
/// An embedding layer acts as a simple lookup table that stores embeddings.
/// This is commonly used to store word embeddings.
#[derive(Debug, Clone)]
pub struct Embedding {
    pub ws: Tensor,
    config: EmbeddingConfig,
}

pub struct EmbeddingConfigFromParts {
    pub sparse: bool,
    pub scale_grad_by_freq: bool,
    pub padding_idx: i64,
}

impl Default for EmbeddingConfigFromParts {
    fn default() -> Self {
        Self {
            sparse: EmbeddingConfig::default().sparse,
            scale_grad_by_freq: EmbeddingConfig::default().scale_grad_by_freq,
            padding_idx: EmbeddingConfig::default().padding_idx,
        }
    }
}

impl Embedding {
    pub fn embedding_dim(&self) -> i64 {
        self.ws.size2().unwrap().1
    }

    pub fn from_parts(ws: Tensor, config: EmbeddingConfigFromParts) -> Self {
        Self {
            ws,
            config: EmbeddingConfig {
                sparse: config.sparse,
                scale_grad_by_freq: config.scale_grad_by_freq,
                padding_idx: config.padding_idx,
                ws_init: super::Init::Const(0.),
            },
        }
    }
}

pub fn embedding<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    num_embeddings: i64,
    embedding_dim: i64,
    config: EmbeddingConfig,
) -> Embedding {
    let vs = vs.borrow();
    Embedding {
        ws: vs.var(
            "embedding",
            &[num_embeddings, embedding_dim],
            config.ws_init,
        ),
        config,
    }
}

impl super::module::Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::embedding(
            &self.ws,
            xs,
            self.config.padding_idx,
            self.config.scale_grad_by_freq,
            self.config.sparse,
        )
    }
}

impl ToDevice for Embedding {
    fn to_device(&self, device: Device) -> Self {
        Self {
            ws: self.ws.to_device(device),
            config: self.config,
        }
    }

    fn device(&self) -> Device {
        self.ws.device()
    }
}
