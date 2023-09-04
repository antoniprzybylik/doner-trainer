use nalgebra as na;

use na::SMatrix;
use na::SVector;

use super::Layer;

pub struct GeLULayer<const SIZE: usize> {
    input: SVector<f64, SIZE>,
    pub signal: SVector<f64, SIZE>,
    chain_element: SMatrix<f64, SIZE, SIZE>,
}

impl<const SIZE: usize> GeLULayer<SIZE>
{
    pub fn new() -> Self {
        Self {
            input: na::zero(),
            signal: na::zero(),
            chain_element: na::zero(),
        }
    }
}

fn gerror(x: f64) -> f64 {
	0.5*x*(1.0 + ((2.0/std::f64::consts::PI).sqrt()*
		          (x + 0.044715*x*x*x)).tanh())
}

fn gerror_derivative(x: f64) -> f64 {
    0.05351611220*x*x*x -
	0.05351611220*
	(0.03567740814*x*x*x +
	 0.7978845608*x).tanh()*
	(0.03567740814*x*x*x +
	 0.7978845608*x).tanh()*x*x*x +
	0.3989422804*x -
	0.3989422804*
	(0.03567740814*x*x*x +
	 0.7978845608*x).tanh()*
	(0.03567740814*x*x*x +
	 0.7978845608*x).tanh()*x +
	0.5000000000 +
	0.5000000000*
	(0.03567740814*x*x*x +
	 0.7978845608*x).tanh()
}

impl<const SIZE: usize> Layer<SIZE, SIZE> for GeLULayer<SIZE> {
    const PARAMS_CNT: usize = 0;

    unsafe fn eval_unchecked(p: &[f64], x: SVector<f64, SIZE>) -> SVector<f64, SIZE> {
        Self::eval(p, x)
    }

    fn eval(_p: &[f64], x: SVector<f64, SIZE>) -> SVector<f64, SIZE> {
        let mut x = x;

        for xi in x.iter_mut() {
            *xi = gerror(*xi);
        }

        x
    }

    fn forward(&mut self, p: &[f64], x: SVector<f64, SIZE>) -> SVector<f64, SIZE> {
        self.input = x.clone();
        self.signal = Self::eval(p, x);

        self.signal.clone()
    }

    fn backward(&mut self, _p: &[f64]) {
        self.chain_element = na::zero();
        for i in 0..SIZE {
            self.chain_element[(i, i)] = gerror_derivative(self.input[i]);
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
        
        let y = GeLULayer::<2>::eval(&[], x);
        assert_eq!(y, na::vector![gerror(1f64), gerror(2f64)]);
    }

    #[test]
    fn test_forward() {
        let x = na::vector![2., -3.];
        let mut layer = GeLULayer::<2>::new();
        
        let y = layer.forward(&[], x);
        assert_eq!(y, na::vector![gerror(2f64), gerror(-3f64)]);
    }

    #[test]
    fn test_backward() {
        let x = na::vector![1., 2.];
        let mut layer = GeLULayer::<2>::new();
        
        let _ = layer.forward(&[], x);
        layer.backward(&[]);

        assert_eq!(*layer.chain_element(),
                   na::matrix![gerror_derivative(1f64), 0f64;
                               0f64, gerror_derivative(2f64)]);
    }

    #[test]
    fn test_chain_end() {
        let x = na::vector![1f64, 2f64];
        let layer = GeLULayer::<2>::new();

        assert_eq!(layer.chain_end(&x).ncols(), 0);
    }
}
