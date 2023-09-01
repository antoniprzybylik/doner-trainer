use nalgebra as na;

use na::SMatrix;
use na::SVector;

use super::Layer;

pub struct SigmaLayer<const SIZE: usize> {
    pub signal: SVector<f64, SIZE>,
    chain_element: SMatrix<f64, SIZE, SIZE>,
}

impl<const SIZE: usize> SigmaLayer<SIZE>
{
    pub fn new() -> Self {
        Self {
            signal: na::zero(),
            chain_element: na::zero(),
        }
    }
}

fn sigma(x: f64) -> f64 {
    ((x / 2.).tanh() + 1.) / 2.
}

fn sigma_d(s: f64) -> f64 {
    s*(1. - s)
}

impl<const SIZE: usize> Layer<SIZE, SIZE> for SigmaLayer<SIZE> {
    const PARAMS_CNT: usize = 0;

    unsafe fn eval_unchecked(p: &[f64], x: SVector<f64, SIZE>) -> SVector<f64, SIZE> {
        Self::eval(p, x)
    }

    fn eval(_p: &[f64], x: SVector<f64, SIZE>) -> SVector<f64, SIZE> {
        let mut x = x;

        for xi in x.iter_mut() {
            *xi = sigma(*xi);
        }

        x
    }

    fn forward(&mut self, p: &[f64], x: SVector<f64, SIZE>) -> SVector<f64, SIZE> {
        self.signal = Self::eval(p, x);

        self.signal.clone()
    }

    fn backward(&mut self, _p: &[f64]) {
        self.chain_element = na::zero();
        for i in 0..SIZE {
            self.chain_element[(i, i)] = sigma_d(self.signal[i]);
        }
    }

    fn chain_element(&self) -> &SMatrix<f64, SIZE, SIZE> {
        &self.chain_element
    }

    fn chain_end(&self, _x: &SVector<f64, SIZE>) ->
        SMatrix<f64, SIZE, { Self::PARAMS_CNT }>
    {
        na::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval() {
        let x = na::vector![1., 2.];
        
        let y = SigmaLayer::<2>::eval(&[], x);
        assert_eq!(y, na::vector![sigma(1f64), sigma(2f64)]);
    }

    #[test]
    fn test_forward() {
        let x = na::vector![2., -3.];
        let mut layer = SigmaLayer::<2>::new();
        
        let y = layer.forward(&[], x);
        assert_eq!(y, na::vector![sigma(2f64), sigma(-3f64)]);
    }

    #[test]
    fn test_backward() {
        let x = na::vector![1., 2.];
        let mut layer = SigmaLayer::<2>::new();
        
        let _ = layer.forward(&[], x);
        layer.backward(&[]);

        assert_eq!(*layer.chain_element(),
                   na::matrix![sigma_d(sigma(1f64)), 0f64;
                               0f64, sigma_d(sigma(2f64))]);
    }

    #[test]
    fn test_chain_end() {
        let x = na::vector![1f64, 2f64];
        let layer = SigmaLayer::<2>::new();

        assert_eq!(layer.chain_end(&x).ncols(), 0);
    }
}
