//! A linear fully-connected layer.
use crate::{AsView, Device, Tensor, ToDevice};
use std::borrow::Borrow;

/// Configuration for a linear layer.
#[derive(Debug, Clone, Copy)]
pub struct LinearConfig {
    pub ws_init: super::Init,
    pub bs_init: Option<super::Init>,
}

impl Default for LinearConfig {
    fn default() -> Self {
        LinearConfig {
            ws_init: super::Init::KaimingUniform,
            bs_init: None,
        }
    }
}

/// A linear fully-connected layer.
#[derive(Debug, Clone)]
pub struct Linear {
    pub ws: Tensor,
    pub bs: Tensor,
}

/// Creates a new linear layer.
pub fn linear<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    c: LinearConfig,
) -> Linear {
    let vs = vs.borrow();
    let bs_init = c.bs_init.unwrap_or_else(|| {
        let bound = 1.0 / (in_dim as f64).sqrt();
        super::Init::Uniform {
            lo: -bound,
            up: bound,
        }
    });
    Linear {
        ws: vs.var("weight", &[out_dim, in_dim], c.ws_init),
        bs: vs.var("bias", &[out_dim], bs_init),
    }
}

impl super::module::Module for Linear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.matmul(&self.ws.tr()) + &self.bs
    }
}

impl ToDevice for Linear {
    fn to_device(&self, device: Device) -> Self {
        Self {
            ws: self.ws.to_device(device),
            bs: self.bs.to_device(device),
        }
    }

    fn device(&self) -> Device {
        self.ws.device()
    }
}

impl AsView for Linear {
    fn as_view(&self) -> Self {
        Self {
            ws: self.ws.as_view(),
            bs: self.bs.as_view(),
        }
    }
}
