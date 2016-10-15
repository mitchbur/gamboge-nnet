using System;
using System.Collections.Generic;
using System.Linq;

namespace Gamboge
{
    /// <summary>
    /// artificial neural network
    /// </summary>
    public class NeuralNetwork
    {
        /// <summary>
        /// constructor
        /// </summary>
        /// <param name="n">number of network inputs</param>
        /// <param name="m">number of hidden layer units</param>
        /// <param name="wts">network weights</param>
        public NeuralNetwork(int n, int m, IEnumerable<float> wts)
        {
            input_count = n;
            hidden_count = m;
            weights = wts;
            outtransform = NeuralNetworkMath.logistic_output;
        }

        /// <summary>
        /// constructor
        /// </summary>
        /// <param name="n">number of network inputs</param>
        /// <param name="m">number of hidden layer units</param>
        /// <param name="wts">network weights</param>
        /// <param name="otransform">output transfer function</param>
        public NeuralNetwork(int n, int m, IEnumerable<float> wts, Func<float, float> otransform)
        {
            input_count = n;
            hidden_count = m;
            weights = wts;
            outtransform = otransform;
        }

        private int input_count;
        private int hidden_count;
        private IEnumerable<float> weights;
        Func<float, float> outtransform;

        public float Compute(IEnumerable<float> values)
        {
            return NeuralNetworkMath.ann_compute(values, weights, input_count, hidden_count, outtransform);
        }
    }

    public static class NeuralNetworkMath
    {
        public static float logistic_output(float x)
        {
            return 1.0F / (1.0F + (float)Math.Exp(-x));
        }

        public static float linear_output(float x)
        {
            return x;
        }

        /// <summary>
        /// Compute artificial neural network output.
        /// </summary>
        /// <param name="values">input value sequence</param>
        /// <param name="weights">neural network weights</param>
        /// <param name="n">count of neural network inputs</param>
        /// <param name="m">neural network output value</param>
        /// <param name="outtransform">output transfer function for hidden and output units</param>
        /// <returns>neural network output value</returns>
        /// <remarks>
        /// <para>
        /// Computes output of an artificial neural network with <paramref name="n"/> inputs,
        /// <paramref name="m"/> hidden layer units and one output unit.
        /// <paramref name="values"/> are the network inputs.
        /// </para><para>
        /// For each unit the inner product of its inputs and weights is added to
        /// a bias weight. This result is then fed into a logistic function to
        /// produce the unit's output.</para>
        /// <para>y = L( bias + &lang; x, w &rang; )
        /// </para><para>
        /// Each hidden layer unit has one bias input and @a n inputs connected to the n
        /// network inputs. For each of the unit's inputs there is a corresponding weight. The
        /// first 1 + n values in weights are the first hidden layer
        /// unit's bias and input weights; the first value is the bias and the remaining n
        /// weights apply to the n network inputs in order. The next 1 + n values
        /// in weights are the second hidden layer unit's weights, etc.
        /// </para><para>
        /// When m is greater than 0, the one output layer unit has one bias input and
        /// m inputs connected to the m hidden layer outputs. The output layer
        /// unit has 1+m weights and these are the last values in the weights sequence.
        /// </para><para>
        /// If m equals 0 then there is no hidden layer and the network inputs
        /// connect directly to the one output unit. In this case the first 1 + n 
        /// values in weights apply to the output unit;
        /// the first value is the bias and the remaining n weights apply
        /// to the n network inputs in order.
        /// </para>
        /// </remarks>
        public static float ann_compute(
            IEnumerable<float> values,
            IEnumerable<float> weights,
            int n,
            int m,
            Func<float, float> outtransform )
        {
            int itw = 0;

            if ( m > 0 )
            {
                // compute each of the hidden layer outputs
                List<float> hidden_out = new List<float>(m);

                for (int hk = 0; hk < m; ++hk)
                {
                    float bias = weights.Skip( itw ).First();
                    ++itw;
                    float oh = bias + weights.Skip(itw).Take(n).Zip(values, (w, x) => w * x).Sum();
                    itw += n;
                    hidden_out.Add( outtransform(oh) );
                }

                // feed the hidden layer outputs into the output layer unit
                {
                    float bias = weights.Skip(itw).First();
                    ++itw;
                    float oh = bias + weights.Skip(itw).Take(m).Zip(hidden_out, (w, x) => w * x).Sum();
                    return outtransform(oh);
                }
            }
            else  // m is 0
            {
                // feed the network inputs directly into the output layer unit
                float bias = weights.Skip(itw).First();
                ++itw;
                float oh = bias + weights.Skip(itw).Take(n).Zip(values, (w, x) => w * x).Sum();
                return outtransform(oh);
            }
        }
    }
}
