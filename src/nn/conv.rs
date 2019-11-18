//! N-dimensional convolution layers.
use super::Path;
use crate::{AsView, Device, Tensor, ToDevice};
use failure::{err_msg, Error};
use std::borrow::Borrow;

/// Generic convolution config.
#[derive(Debug, Clone, Copy)]
pub struct ConvConfigND<ND> {
    pub stride: ND,
    pub padding: ND,
    pub dilation: ND,
    pub groups: i64,
    pub bias: bool,
    pub ws_init: super::Init,
    pub bs_init: super::Init,
}

/// Convolution config using the same parameters on all dimensions.
pub type ConvConfig = ConvConfigND<i64>;

impl Default for ConvConfig {
    fn default() -> Self {
        ConvConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
            ws_init: super::Init::KaimingUniform,
            bs_init: super::Init::Const(0.),
        }
    }
}

impl Default for ConvConfigND<[i64; 2]> {
    fn default() -> Self {
        ConvConfigND::<[i64; 2]> {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            bias: true,
            ws_init: super::Init::KaimingUniform,
            bs_init: super::Init::Const(0.),
        }
    }
}

/// The default convolution config without bias.
pub fn no_bias() -> ConvConfig {
    ConvConfig {
        bias: false,
        ..Default::default()
    }
}

// Use const generics when they have landed in stable rust.
/// A N-dimensional convolution layer.
#[derive(Debug, Clone)]
pub struct Conv<ND> {
    pub ws: Tensor,
    pub bs: Option<Tensor>,
    config: ConvConfigND<ND>,
}

#[derive(Debug, Clone, Copy)]
pub struct ConvBuilderConfig<ND> {
    pub stride: ND,
    pub padding: ND,
    pub dilation: ND,
    pub groups: i64,
}

impl<ND> Conv<ND> {
    pub fn from_parts(ws: Tensor, bs: Option<Tensor>, config: ConvBuilderConfig<ND>) -> Result<Self, Error> {
        let bias = bs.is_some();
        Ok(Self {
            ws,
            bs,
            config: ConvConfigND {
                stride: config.stride,
                padding: config.padding,
                dilation: config.dilation,
                groups: config.groups,
                bias,
                ws_init: super::Init::Const(0.),
                bs_init: super::Init::Const(0.),
            },
        })
    }

    pub fn in_dim(&self) -> i64 {
        self.ws.size()[1] * self.config.groups
    }

    pub fn config(&self) -> &ConvConfigND<ND> {
        &self.config
    }
}

pub trait CnnWeights {
    fn ksizes(&self) -> Vec<i64>;
    fn out_dim(&self) -> i64;
    fn padding(&self, padding: Padding) -> Result<Vec<i64>, Error>;
}

impl<ND> CnnWeights for Conv<ND> {
    fn ksizes(&self) -> Vec<i64> {
        self.ws.ksizes()
    }
    fn out_dim(&self) -> i64 {
        self.ws.out_dim()
    }
    fn padding(&self, padding: Padding) -> Result<Vec<i64>, Error> {
        self.ws.padding(padding)
    }
}

impl CnnWeights for Tensor {
    fn ksizes(&self) -> Vec<i64> {
        self.size().split_off(2)
    }
    fn out_dim(&self) -> i64 {
        self.size()[0]
    }
    fn padding(&self, padding: Padding) -> Result<Vec<i64>, Error> {
        padding.padding(self.ksizes())
    }
}

pub enum Padding {
    Same,
    Valid,
}

impl Padding {
    fn padding(self, ksizes: Vec<i64>) -> Result<Vec<i64>, Error> {
        match self {
            Padding::Same => {
                ksizes
                    .into_iter()
                    .map(|x| if x % 2 == 1 {
                        Ok(x / 2)
                    } else {
                        Err(err_msg("Padding::Same requires an odd kernel size."))
                    })
                    .collect::<Result<Vec<_>, _>>()
            }
            Padding::Valid => Ok(vec![0; ksizes.len()]),
        }
    }
}

/// One dimension convolution layer.
pub type Conv1D = Conv<[i64; 1]>;

/// Two dimensions convolution layer.
pub type Conv2D = Conv<[i64; 2]>;

/// Three dimensions convolution layer.
pub type Conv3D = Conv<[i64; 3]>;

/// Creates a new convolution layer for any number of dimensions.
pub fn conv<'a, ND: std::convert::AsRef<[i64]>, T: Borrow<super::Path<'a>>>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    ksizes: ND,
    config: ConvConfigND<ND>,
) -> Conv<ND> {
    let vs = vs.borrow();
    let bs = if config.bias {
        Some(vs.var("bias", &[out_dim], config.bs_init))
    } else {
        None
    };
    let mut weight_size = vec![out_dim, in_dim / config.groups];
    weight_size.extend(ksizes.as_ref().iter());
    let ws = vs.var("weight", weight_size.as_slice(), config.ws_init);
    Conv { ws, bs, config }
}

trait Create: std::convert::AsRef<[i64]> + std::marker::Sized {
    fn make_array(i: i64) -> Self;

    fn conv<'a, T: Borrow<super::Path<'a>>>(
        vs: T,
        in_dim: i64,
        out_dim: i64,
        ksize: i64,
        config: ConvConfig,
    ) -> Conv<Self> {
        let config = ConvConfigND::<Self> {
            stride: Self::make_array(config.stride),
            padding: Self::make_array(config.padding),
            dilation: Self::make_array(config.dilation),
            groups: config.groups,
            bias: config.bias,
            ws_init: config.ws_init,
            bs_init: config.bs_init,
        };
        conv(vs, in_dim, out_dim, Self::make_array(ksize), config)
    }
}

impl Create for [i64; 1] {
    fn make_array(i: i64) -> Self {
        [i]
    }
}

impl Create for [i64; 2] {
    fn make_array(i: i64) -> Self {
        [i, i]
    }
}

impl Create for [i64; 3] {
    fn make_array(i: i64) -> Self {
        [i, i, i]
    }
}

/// Creates a new one dimension convolution layer.
pub fn conv1d<'a, T: Borrow<Path<'a>>>(vs: T, i: i64, o: i64, k: i64, c: ConvConfig) -> Conv1D {
    <[i64; 1]>::conv(vs, i, o, k, c)
}

/// Creates a new two dimension convolution layer.
pub fn conv2d<'a, T: Borrow<Path<'a>>>(vs: T, i: i64, o: i64, k: i64, c: ConvConfig) -> Conv2D {
    <[i64; 2]>::conv(vs, i, o, k, c)
}

/// Creates a new three dimension convolution layer.
pub fn conv3d<'a, T: Borrow<Path<'a>>>(vs: T, i: i64, o: i64, k: i64, c: ConvConfig) -> Conv3D {
    <[i64; 3]>::conv(vs, i, o, k, c)
}

impl super::module::Module for Conv1D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::conv1d(
            &xs,
            &self.ws,
            self.bs.as_ref(),
            &self.config.stride,
            &self.config.padding,
            &self.config.dilation,
            self.config.groups,
        )
    }
}

impl super::module::Module for Conv2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::conv2d(
            &xs,
            &self.ws,
            self.bs.as_ref(),
            &self.config.stride,
            &self.config.padding,
            &self.config.dilation,
            self.config.groups,
        )
    }
}

impl super::module::Module for Conv3D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::conv3d(
            &xs,
            &self.ws,
            self.bs.as_ref(),
            &self.config.stride,
            &self.config.padding,
            &self.config.dilation,
            self.config.groups,
        )
    }
}

impl<ND> ToDevice for Conv<ND>
where
    ConvConfigND<ND>: Clone,
{
    fn to_device(&self, device: Device) -> Self {
        Self {
            ws: self.ws.to_device(device),
            bs: self.bs.as_ref().map(|x| x.to_device(device)),
            config: self.config.clone(),
        }
    }

    fn device(&self) -> Device {
        self.ws.device()
    }
}

impl<ND> AsView for Conv<ND>
where
    ConvConfigND<ND>: Clone,
{
    fn as_view(&self) -> Self {
        Self {
            ws: self.ws.as_view(),
            bs: self.bs.as_ref().map(|x| x.as_view()),
            config: self.config.clone(),
        }
    }
}
