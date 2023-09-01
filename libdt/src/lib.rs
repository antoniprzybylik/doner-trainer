#![feature(generic_const_exprs)]

#[cfg(feature = "macros")]
pub use libdt_macros as macros;

pub mod layer;
pub mod network;
pub mod trainer;
