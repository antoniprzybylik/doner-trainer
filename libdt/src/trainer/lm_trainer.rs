use nalgebra::SVector;
use nalgebra::RowSVector;
use nalgebra::SMatrix;

use super::super::network::Network;
use super::super::trainer::Trainer;

use super::common::cost;
use super::common::apply_step;
use super::common::eval_untouched;

/// Trainer using Levenberg-Marquardt Method.
pub struct LMTrainer<N,
                     const PARAMS_CNT: usize,
                     const NEURONS_IN: usize,
                     const NEURONS_OUT: usize>
where
    N: Network<PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
{
    p: Vec<f64>,
    x_values: Vec<SVector<f64, NEURONS_IN>>,
    d_values: Vec<SVector<f64, NEURONS_OUT>>,
    nn: N,

    lambda: f64,
}

fn net_eval<N,
    const PARAMS_CNT: usize,
    const NEURONS_IN: usize,
    const NEURONS_OUT: usize>(p: &[f64],
    x_values: &[SVector<f64, NEURONS_IN>])
    -> Vec<SVector<f64, NEURONS_OUT>>
where
    N: Network<PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
{
    let mut y_values: Vec<SVector<f64, NEURONS_OUT>> =
        Vec::new();

    for x in x_values.into_iter() {
        y_values.push(N::eval(&p, x.clone()));
    }

    y_values
}

impl<N, const PARAMS_CNT: usize,
     const NEURONS_IN: usize,
     const NEURONS_OUT: usize>
    LMTrainer<N, PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
where
    N: Network<PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
{
    fn choose_lm_step(
         &mut self,
         h: SMatrix<f64, PARAMS_CNT, PARAMS_CNT>,
         g: RowSVector<f64, PARAMS_CNT>)
        -> RowSVector<f64, PARAMS_CNT>
    where
        N: Network<PARAMS_CNT, NEURONS_IN, NEURONS_OUT>,
    {
        let current_cost = eval_untouched::<
            N, PARAMS_CNT, NEURONS_IN, NEURONS_OUT>(
            &mut self.p, &nalgebra::zero(),
            self.x_values.as_slice(), self.d_values.as_slice());
    
        loop {
            let mut m = h.clone();
            for i in 0..m.ncols() {
                m[(i, i)] += self.lambda*m[(i, i)];
            }
            let m = m.try_inverse();
            let step = match m {
                Some(m) => -(&m * g.transpose()).transpose(),
                None => return super::common::choose_step::<N,
                    PARAMS_CNT, NEURONS_IN, NEURONS_OUT>(
                        &mut self.p, self.x_values.as_slice(),
                        self.d_values.as_slice(), -g),
            };

            if eval_untouched::<
                N, PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
                (&mut self.p, &step,
                 self.x_values.as_slice(), self.d_values.as_slice())
                < current_cost {

                let new_lambda = self.lambda / 9f64;
                if new_lambda < 1e-12 {
                    self.lambda = 1e-12;
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
        (RowSVector<f64, PARAMS_CNT>,
         SMatrix<f64, NEURONS_OUT, PARAMS_CNT>)
    {
            let mut jm_sum: SMatrix<f64, NEURONS_OUT, PARAMS_CNT> =
                nalgebra::zero();
            let mut g_sum: RowSVector<f64, PARAMS_CNT> =
                nalgebra::zero();
    
            for i in 0..self.x_values.len() {
                let x = &self.x_values[i];
                let d = &self.d_values[i];
    
                let y = self.nn.forward(&self.p, x.clone());
                self.nn.backward(&self.p);
                let jm = self.nn.jacobian(x);

                g_sum += (y - d).transpose() * jm;
                jm_sum += jm;
            }
            
            (g_sum, jm_sum)
    }
}

impl<N, const PARAMS_CNT: usize,
     const NEURONS_IN: usize,
     const NEURONS_OUT: usize> 
     Trainer<N, PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
    for LMTrainer<N, PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
where
    N: Network<PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
{
    fn new(nn: N, p: Vec<f64>,
           x_values: Vec<SVector<f64, NEURONS_IN>>,
           d_values: Vec<SVector<f64, NEURONS_OUT>>) -> Self
    {
        assert_eq!(p.len(), PARAMS_CNT);
        assert_eq!(x_values.len(),
                   d_values.len());

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
        let y_values = net_eval::<N, PARAMS_CNT,
                                  NEURONS_IN, NEURONS_OUT>(
                                self.p.as_slice(),
                                self.x_values.as_slice());
        let y_values = y_values.as_slice();

        cost(y_values, self.d_values.as_slice())
    }

    fn grad(&mut self) -> RowSVector<f64, PARAMS_CNT>
    {
            let mut grad_sum: RowSVector<f64, PARAMS_CNT> =
                nalgebra::zero();
    
            for i in 0..self.x_values.len() {
                let x = &self.x_values[i];
                let d = &self.d_values[i];
    
                let y = self.nn.forward(&self.p, x.clone());
                self.nn.backward(&self.p);
                let jm = self.nn.jacobian(x);
                let g = (y - d).transpose() * jm;
    
                grad_sum += g;
            }
    
            grad_sum
    }

    fn grad_norm(&mut self) -> f64 {
        let direction = self.grad();

        direction.norm()
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
    use super::super::super::network::Network;
    use super::super::super::macros::neural_network;

    use super::*;

    #[neural_network]
    struct NiceNetwork {
        layers: (LinLayer::<1, 10>,
                 SigmaLayer::<10>,
                 LinLayer::<10, 1>)
    }

    #[test]
    fn test_grad() {
        let mut x_values: Vec<SVector<f64, 1>> =
            vec![nalgebra::vector![3.11f64],
                 nalgebra::vector![4.15f64]];
        let mut d_values: Vec<SVector<f64, 1>> =
            vec![nalgebra::vector![1.78215f64],
                 nalgebra::vector![8.18725f64]];
    
        let mut rng = rand::thread_rng();
        let mut p: Vec<f64> = Vec::new();
        for _ in 0..NiceNetwork::PARAMS_CNT {
            p.push(rng.gen_range(0.0..1.0));
        }
    
        let nn = NiceNetwork::new();
        let mut trainer = LMTrainer::new(
            nn, p, x_values, d_values);

        let g1 = trainer.grad();
        let (g2, _) = trainer.grad_and_jacobian();

        assert_eq!(g1.len(), g2.len());
        for i in 0..g1.len() {
            assert_float_eq!(g1[i], g2[i], abs <= 0.000_000_000_1);
        }
    }
}
