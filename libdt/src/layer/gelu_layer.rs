use nalgebra as na;

use na::DMatrix;
use na::DVector;
use na::Matrix;
use na::Vector;
use na::base::dimension as dim;
use na::base::VecStorage;

use super::Layer;

pub struct GeLULayer<const SIZE: usize> {
    input: Vector::<f64, dim::Dyn,
                    VecStorage::<f64, dim::Dyn, dim::U1>>,
    pub signal: Vector::<f64, dim::Dyn,
                         VecStorage::<f64, dim::Dyn, dim::U1>>,
    chain_element: Matrix::<f64, dim::Dyn, dim::Dyn,
                            VecStorage::<f64, dim::Dyn, dim::Dyn>>,
}

impl<const SIZE: usize> GeLULayer<SIZE>
{
    pub fn new() -> Self {
        Self {
            input: Vector::from_element_generic(dim::Dyn(Self::NEURONS_IN),
                                                 dim::U1, 0f64),
            signal: Vector::from_element_generic(dim::Dyn(Self::NEURONS_OUT),
                                                 dim::U1, 0f64),
            chain_element: Matrix::from_element_generic(dim::Dyn(Self::NEURONS_OUT),
                                                        dim::Dyn(Self::NEURONS_IN), 0f64),
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

impl<const SIZE: usize> Layer for GeLULayer<SIZE> {
    const PARAMS_CNT: usize = 0;
    const NEURONS_IN: usize = SIZE;
    const NEURONS_OUT: usize = SIZE;

    unsafe fn eval_unchecked(p: &[f64], x: DVector<f64>) -> DVector<f64> {
        Self::eval(p, x)
    }

    fn eval(_p: &[f64], x: DVector<f64>) -> DVector<f64> {
        assert_eq!(x.len(), Self::NEURONS_IN);

        let mut x = x;
        for xi in x.iter_mut() {
            *xi = gerror(*xi);
        }

        x
    }

    fn forward(&mut self, p: &[f64], x: DVector<f64>) -> DVector<f64> {
        assert_eq!(p.len(), Self::PARAMS_CNT);
        assert_eq!(x.len(), Self::NEURONS_IN);

        self.input = x.clone();
        self.signal = Self::eval(p, x);
        self.signal.clone()
    }

    fn backward(&mut self, _p: &[f64]) {
        self.chain_element = DMatrix::from_element_generic(
            dim::Dyn(Self::NEURONS_OUT), dim::Dyn(Self::NEURONS_IN), 0f64);
        for i in 0..SIZE {
            self.chain_element[(i, i)] = gerror_derivative(self.input[i]);
        }
    }

    fn chain_element(&self) -> &DMatrix<f64> {
        &self.chain_element
    }

    fn chain_end(&self, _x: &DVector<f64>) -> DMatrix<f64>
    {
        DMatrix::from_element_generic(
            dim::Dyn(Self::NEURONS_OUT),
            dim::Dyn(Self::PARAMS_CNT), 0f64)
    }

    fn default_initial_params() -> Vec<f64> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval() {
        let x = DVector::from_column_slice(
            na::vector![1f64, 2f64].as_slice());
        
        let y = GeLULayer::<2>::eval(&[], x);
        assert_eq!(y, na::vector![gerror(1f64), gerror(2f64)]);
    }

    #[test]
    fn test_forward() {
        let x = DVector::from_column_slice(
            na::vector![2f64, -3f64].as_slice());
        let mut layer = GeLULayer::<2>::new();
        
        let y = layer.forward(&[], x);
        assert_eq!(y, na::vector![gerror(2f64), gerror(-3f64)]);
    }

    #[test]
    fn test_backward() {
        let x = DVector::from_column_slice(
            na::vector![1f64, 2f64].as_slice());
        let mut layer = GeLULayer::<2>::new();
        
        let _ = layer.forward(&[], x);
        layer.backward(&[]);

        assert_eq!(*layer.chain_element(),
                   na::matrix![gerror_derivative(1f64), 0f64;
                               0f64, gerror_derivative(2f64)]);
    }

    #[test]
    fn test_chain_end() {
        let x = DVector::from_column_slice(
            na::vector![1f64, 2f64].as_slice());
        let layer = GeLULayer::<2>::new();

        assert_eq!(layer.chain_end(&x).ncols(), 0);
    }
}
