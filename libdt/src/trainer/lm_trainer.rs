use nalgebra::DVector;
use nalgebra::RowDVector;
use nalgebra::DMatrix;
use nalgebra::Matrix;
use nalgebra::base::dimension as dim;

use super::super::network::Network;
use super::super::trainer::Trainer;

use super::common::cost;
use super::common::apply_step;
use super::common::eval_untouched;

/// Trainer using Levenberg-Marquardt Method.
pub struct LMTrainer<N: Network>
{
    p: Vec<f64>,
    x_values: Vec<DVector<f64>>,
    d_values: Vec<DVector<f64>>,
    nn: N,

    lambda: f64,
}

fn net_eval<N: Network>(p: &[f64], x_values: &[DVector<f64>])
    -> Vec<DVector<f64>>
{
    let mut y_values: Vec<DVector<f64>> = Vec::new();
    for x in x_values.into_iter() {
        assert_eq!(x.len(), N::NEURONS_IN);
        y_values.push(N::eval(&p, x.clone()));
    }

    y_values
}

impl<N: Network> LMTrainer<N> {
    fn choose_lm_step(
         &mut self, h: DMatrix<f64>, g: RowDVector<f64>)
        -> RowDVector<f64>
    {
        assert_eq!(h.nrows(), N::PARAMS_CNT);
        assert_eq!(h.ncols(), N::PARAMS_CNT);
        assert_eq!(g.len(), N::PARAMS_CNT);

        let current_cost = eval_untouched::<N>(
            &mut self.p,
            &Matrix::from_element_generic(
                dim::U1, dim::Dyn(N::PARAMS_CNT), 0f64),
            self.x_values.as_slice(),
            self.d_values.as_slice());
    
        loop {
            let mut m = h.clone();
            for i in 0..m.ncols() {
                m[(i, i)] += self.lambda*m[(i, i)];
            }
            let m = m.try_inverse();
            let step = match m {
                Some(m) => -(&m * g.transpose()).transpose(),
                None => return super::common::choose_step::<N>(
                        &mut self.p, self.x_values.as_slice(),
                        self.d_values.as_slice(), -g),
            };
            
            let rho = (current_cost - eval_untouched::<N>(
                &mut self.p, &step,
                self.x_values.as_slice(),
                self.d_values.as_slice())) /
                (step.clone() *
                 (self.lambda*
                  DMatrix::from_diagonal(&h.diagonal())*
                  step.transpose() - g.transpose())).norm();

            if rho > 0.1f64 {
                let new_lambda = self.lambda / 9f64;
                if new_lambda < 1e-7 {
                    self.lambda = 1e-7;
                } else {
                    self.lambda = new_lambda;
                }

                break step;
            } else {
                self.lambda *= 11f64;
            }
        }
    }

    fn grad_and_jacobian(&mut self) ->
        (RowDVector<f64>, DMatrix<f64>)
    {
            let mut jm_sum: DMatrix<f64> =
                Matrix::from_element_generic(
                    dim::Dyn(N::NEURONS_OUT),
                    dim::Dyn(N::PARAMS_CNT), 0f64);
            let mut g_sum: RowDVector<f64> =
                Matrix::from_element_generic(
                    dim::U1,
                    dim::Dyn(N::PARAMS_CNT), 0f64);
    
            for i in 0..self.x_values.len() {
                let x = &self.x_values[i];
                let d = &self.d_values[i];
    
                let y = self.nn.forward(&self.p, x.clone());
                self.nn.backward(&self.p);
                let jm = self.nn.jacobian(x);

                g_sum += 2f64 * (y - d).transpose() * jm.clone();
                jm_sum += jm;
            }
            
            (g_sum, jm_sum)
    }
}

impl<N: Network> Trainer<N> for LMTrainer<N> {
    fn new(nn: N, p: Vec<f64>,
           x_values: Vec<DVector<f64>>,
           d_values: Vec<DVector<f64>>) -> Self
    {
        assert_eq!(p.len(), N::PARAMS_CNT);
        assert_eq!(x_values.len(), d_values.len());

        LMTrainer {
            p,
            x_values,
            d_values,
            nn,

            lambda: 0.1f64,
        }
    }

    fn make_step(&mut self) {
        let (g, jm) = self.grad_and_jacobian();
        let h = jm.transpose()*jm;

        let step = self.choose_lm_step(h, g);
        apply_step(&mut self.p, &step);
    }

    fn cost(&self) -> f64 {
        let y_values = net_eval::<N>(
            self.p.as_slice(), self.x_values.as_slice());
        let y_values = y_values.as_slice();

        cost(y_values, self.d_values.as_slice())
    }

    fn grad(&mut self) -> RowDVector<f64>
    {
            let mut grad_sum: RowDVector<f64> =
                Matrix ::from_element_generic(
                    dim::U1, dim::Dyn(N::PARAMS_CNT), 0f64);
    
            for i in 0..self.x_values.len() {
                let x = &self.x_values[i];
                let d = &self.d_values[i];
    
                let y = self.nn.forward(&self.p, x.clone());
                self.nn.backward(&self.p);
                let jm = self.nn.jacobian(x);
                let g = 2f64 * (y - d).transpose() * jm;
    
                grad_sum += g;
            }
    
            grad_sum
    }

    fn grad_norm(&mut self) -> f64 {
        self.grad().norm()
    }

    fn params(&self) -> &[f64] {
        &self.p
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use float_eq::assert_float_eq;
    use super::super::super::layer::Layer;
    use super::super::super::layer::LinLayer;
    use super::super::super::layer::SigmaLayer;
    use super::super::super::layer::SumLayer;
    use super::super::super::network::Network;
    use libdt_macros::neural_network;

    use super::*;

    #[neural_network]
    struct Test1Network {
        layers: (LinLayer::<1, 10>,
                 SigmaLayer::<10>,
                 LinLayer::<10, 1>)
    }

    #[test]
    fn test_grad_1() {
        let x_values: Vec<DVector<f64>> =
            vec![DVector::from_column_slice(
                     nalgebra::vector![3.11f64].as_slice()),
                 DVector::from_column_slice(
                     nalgebra::vector![4.15f64].as_slice())];
        let d_values: Vec<DVector<f64>> =
            vec![DVector::from_column_slice(
                     nalgebra::vector![1.78215f64].as_slice()),
                 DVector::from_column_slice(
                     nalgebra::vector![8.18725f64].as_slice())];
    
        let mut rng = rand::thread_rng();
        let mut p: Vec<f64> = Vec::new();
        for _ in 0..Test1Network::PARAMS_CNT {
            p.push(rng.gen_range(0.0..1.0));
        }
    
        let nn = Test1Network::new();
        let mut trainer = LMTrainer::new(
            nn, p, x_values, d_values);

        let g1 = trainer.grad();
        let (g2, _) = trainer.grad_and_jacobian();

        assert_eq!(g1.len(), g2.len());
        for i in 0..g1.len() {
            assert_float_eq!(g1[i], g2[i], abs <= 0.000_000_000_1);
        }
    }

    #[neural_network]
    struct Test2Network {
        layers: (SumLayer::<1, 1>,)
    }

    #[test]
    fn test_grad_2() {
        let x_values: Vec<DVector<f64>> =
            vec![DVector::from_column_slice(
                     nalgebra::vector![3f64].as_slice())];
        let d_values: Vec<DVector<f64>> =
            vec![DVector::from_column_slice(
                     nalgebra::vector![1f64].as_slice())];
    
        let p: Vec<f64> = vec![2f64];
    
        let nn = Test2Network::new();
        let mut trainer = LMTrainer::new(
            nn, p, x_values, d_values);

        let g = trainer.grad();

        assert_eq!(g, nalgebra::vector![30f64]);
    }

    #[neural_network]
    struct Test3Network {
        layers: (SumLayer::<2, 2>,)
    }

    #[test]
    fn test_grad_3() {
        let x_values: Vec<DVector<f64>> =
            vec![DVector::from_column_slice(
                     nalgebra::vector![3f64, -1.5f64].as_slice()),
                 DVector::from_column_slice(
                     nalgebra::vector![0f64, 0f64].as_slice())];
        let d_values: Vec<DVector<f64>> =
            vec![DVector::from_column_slice(
                     nalgebra::vector![1f64, 2f64].as_slice()),
                 DVector::from_column_slice(
                     nalgebra::vector![2f64, 0f64].as_slice())];
    
        let p: Vec<f64> = vec![2f64, 2f64, 1f64, 0f64];
    
        let nn = Test3Network::new();
        let mut trainer = LMTrainer::new(
            nn, p, x_values, d_values);

        let g = trainer.grad();

        assert_eq!(g.transpose(),
                   nalgebra::vector![21f64, -10.5f64,
                                     24f64, -12f64]);
    }
}
