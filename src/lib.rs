#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate failure;
extern crate libc;
extern crate zip;

pub mod data;

mod wrappers;
pub use wrappers::device::{Cuda, Device, ToDevice};
pub use wrappers::jit::{CModule, IValue};
pub use wrappers::kind::Kind;
pub use wrappers::manual_seed;
pub use wrappers::scalar::Scalar;
pub use wrappers::AsView;

mod tensor;
pub use tensor::{
    index, no_grad, no_grad_guard, IndexOp, SelectIndex, NewAxis, NoGradGuard, Reduction, Tensor, TensorIndexer,
};

pub mod nn;
pub mod vision;

pub mod kind {
    pub(crate) use super::wrappers::kind::T;
    pub use super::wrappers::kind::*;
}

pub fn init_all() {
    unsafe { torch_sys::at_init_all() }
}
