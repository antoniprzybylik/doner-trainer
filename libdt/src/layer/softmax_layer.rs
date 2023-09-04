use nalgebra as na;
use std::cmp::Ordering;

use na::SMatrix;
use na::SVector;

use super::Layer;

pub struct SoftMaxLayer<const SIZE: usize> {
    pub signal: SVector<f64, SIZE>,
    chain_element: SMatrix<f64, SIZE, SIZE>,
}

impl<const SIZE: usize> SoftMaxLayer<SIZE>
{
    pub fn new() -> Self {
        Self {
            signal: na::zero(),
            chain_element: na::zero(),
        }
    }
}

fn softmax_d<const SIZE: usize>(signal: &SVector<f64, SIZE>, i: usize, j: usize) -> f64 {
    (if i == j { signal[i] } else { 0f64 }) -
    signal[i] * signal[j]
}

impl<const SIZE: usize> Layer<SIZE, SIZE> for SoftMaxLayer<SIZE> {
    const PARAMS_CNT: usize = 0;

    unsafe fn eval_unchecked(p: &[f64], x: SVector<f64, SIZE>) -> SVector<f64, SIZE> {
        Self::eval(p, x)
    }

    fn eval(_p: &[f64], x: SVector<f64, SIZE>) -> SVector<f64, SIZE> {
        let mut x = x;

        let mut max_elem: f64 = 0f64;
        for xi in x.iter_mut() {
            match (*xi).partial_cmp(&max_elem) {
                Some(Ordering::Greater) => { max_elem = *xi; }
                _ => {}
            }
        }

        let mut layer_sum: f64 = 0f64;
        for xi in x.iter_mut() {
            *xi = (*xi - max_elem).exp();
            layer_sum += *xi;
        }

        for xi in x.iter_mut() {
            *xi /= layer_sum;
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
            for j in 0..SIZE {
                self.chain_element[(i, j)] = softmax_d(&self.signal, i, j);
            }
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
    use float_eq::assert_float_eq;
    use super::*;

    #[test]
    fn test_eval() {
        let x = na::vector![1., 2.];
        
        let y = SoftMaxLayer::<2>::eval(&[], x);
        assert_eq!(y.len(), 2);
        assert_float_eq!(y[0], 0.2689414214f64, abs <= 0.000_000_000_1);
        assert_float_eq!(y[1], 0.7310585786f64, abs <= 0.000_000_000_1);
    }

    #[test]
    fn test_forward() {
        let x = na::vector![2., -3.];
        let mut layer = SoftMaxLayer::<2>::new();
        
        let y = layer.forward(&[], x);
        assert_eq!(y.len(), 2);
        assert_float_eq!(y[0], 0.9933071491f64, abs <= 0.000_000_000_1);
        assert_float_eq!(y[1], 0.006692850924f64, abs <= 0.000_000_000_1);
    }

    #[test]
    fn test_backward() {
        let x = na::vector![1., 2.];
        let mut layer = SoftMaxLayer::<2>::new();
        
        let _ = layer.forward(&[], x);
        layer.backward(&[]);

        let chain_element = layer.chain_element;
        assert_eq!(chain_element.nrows(), 2);
        assert_eq!(chain_element.ncols(), 2);
        assert_float_eq!(chain_element[(0, 0)], 0.2689414214f64*(1f64 - 0.2689414214f64), abs <= 0.000_000_000_1);
        assert_float_eq!(chain_element[(0, 1)], -0.2689414214f64*0.7310585786f64, abs <= 0.000_000_000_1);
        assert_float_eq!(chain_element[(1, 0)], -0.7310585786f64*0.2689414214f64, abs <= 0.000_000_000_1);
        assert_float_eq!(chain_element[(1, 1)], 0.7310585786f64*(1f64 - 0.7310585786f64), abs <= 0.000_000_000_1);
    }

    #[test]
    fn test_chain_end() {
        let x = na::vector![1f64, 2f64];
        let layer = SoftMaxLayer::<2>::new();

        assert_eq!(layer.chain_end(&x).ncols(), 0);
    }
}
