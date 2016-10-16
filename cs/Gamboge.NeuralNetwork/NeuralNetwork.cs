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
        /// <param name="n">network inputs count</param>
        /// <param name="m">hidden-layer units count</param>
        /// <param name="k">network outputs count</param>
        /// <param name="wts">network weights</param>
        public NeuralNetwork(int n, int m, int k, IEnumerable<float> wts)
        {
            input_count = n;
            hidden_count = m;
            output_count = k;
            weights = wts;
            outtransform = NeuralNetworkMath.logistic_output;
        }

        /// <summary>
        /// constructor
        /// </summary>
        /// <param name="n">network inputs count</param>
        /// <param name="m">hidden-layer units count</param>
        /// <param name="k">network outputs count</param>
        /// <param name="wts">network weights</param>
        /// <param name="otransform">output transfer function</param>
        public NeuralNetwork(int n, int m, int k, IEnumerable<float> wts, Func<float, float> otransform)
        {
            input_count = n;
            hidden_count = m;
            output_count = k;
            weights = wts;
            outtransform = otransform;
        }

        private int input_count;
        private int hidden_count;
        private int output_count;
        private IEnumerable<float> weights;
        Func<float, float> outtransform;

        public IEnumerable<float> Compute(IEnumerable<float> values)
        {
            return NeuralNetworkMath.CalculateNetwork(values, weights, 
                input_count, hidden_count, output_count, outtransform);
        }
    }

    public static class NeuralNetworkMath
    {
        /// <summary>
        /// logistic output function
        /// </summary>
        /// <param name="x">argument</param>
        /// <returns>value y = 1 / ( 1 + e^{-x} )</returns>
        public static float logistic_output(float x)
        {
            return 1.0F / (1.0F + (float)Math.Exp(-x));
        }

        /// <summary>
        /// linear output function
        /// </summary>
        /// <param name="x">argument</param>
        /// <returns>value y = x (identity operator)</returns>
        public static float linear_output(float x)
        {
            return x;
        }

        /// <summary>
        /// Compute artificial neural network output.
        /// </summary>
        /// <param name="inputs">input value sequence</param>
        /// <param name="weights">weights sequence</param>
        /// <param name="nx">input count</param>
        /// <param name="nh">hidden-layer count</param>
        /// <param name="ny">output count</param>
        /// <param name="unaryop">output transform for neural units</param>
        /// <returns>neural network output value</returns>
        /// <remarks>
        /// <para>
        /// Computes feed-forward artificial neural network outputs. The network has 
        /// <c>nx</c> inputs, <c>nh</c> hidden-layer units and <c>ny</c> output units.
        /// Network input values are read from <c>inputs</c> sequence.  Bias and input 
        /// weights for the hidden-layer and output-layer units are read from 
        /// <c>weights</c>, where the number of weights v varies based upon <c>nx</c>,
        /// <c>nh</c> and <c>ny</c>. <c>unaryop</c> is applied at the output of each 
        /// hidden-layer and output-layer unit in the network.
        /// </para><para>
        /// a neural unit output is:
        ///       y = U( bias + 〈 x, w 〉 )
        /// <c>unaryop</c> U applied to the sum of the unit's bias and
        /// the inner product of the unit's inputs and weights.
        /// </para><para>
        /// If <c>nh</c> is greater than 0, the network inputs feed the hidden-layer
        /// units and the hidden-layer units feed the output-layer units. In this 
        /// case the weights sequence starts with blocks for each of the hidden-layer
        /// units followed by blocks for each of the output-layer units. Each 
        /// hidden-layer block consists of 1 + <c>nx</c> values; the first value
        /// is the bias for the unit and the remainder are weights for each of the
        /// network inputs. Each output-layer block consists of 1 + <c>nh</c> values;
        /// the first value is the bias for the unit and the remainder are weights 
        /// for each of the hidden-layer unit outputs. The number of weights is:
        ///    v = nh * ( 1 + nx ) + ny * ( 1 + nh ).
        /// </para><para>
        /// If <c>nh</c> equals 0, there are no hidden-layer units and the network
        /// inputs directly feed the output-layer units.
        /// In this case the weights sequence consists of blocks for the output-layer
        /// units.
        /// Each output-layer block consists of 1 + <c>nx</c> values; the first value
        /// is the bias for the unit and the remainder are weights for each of the
        /// network inputs.
        /// The number of weights is:
        ///    v = ny * ( 1 + nx ).
        /// </para>
        /// </remarks>
        public static IEnumerable<float> CalculateNetwork(
            IEnumerable<float> values,
            IEnumerable<float> weights,
            int nx,
            int nh,
            int ny,
            Func<float, float> unaryop )
        {
            if ( nh > 0 )
            {
                // compute each of the hidden layer outputs

                IEnumerable<float> hidden_out = Enumerable.Range(0, nh).Select(
                    hk =>
                       {
                           int sk = hk * (1 + nx);
                           var oh = weights.Skip(sk).First();  // the bias
                           oh += values.Zip(weights.Skip(sk + 1), (x,w) => w * x).Sum();
                           return unaryop( oh );
                       } );

                // feed the hidden layer outputs into the output layer unit
                int owts_begin = nh * (1 + nx);
                IEnumerable<float> result = Enumerable.Range(0, ny).Select(
                    hy =>
                        {
                            int sk = owts_begin + hy * (1 + nh);
                            var oy = weights.Skip(sk).First();  // the bias
                            oy += hidden_out.Zip(weights.Skip(sk + 1), (x,w) => w * x).Sum();
                            return unaryop(oy);
                        });
                return result;
            }
            else  // m is 0
            {
                // feed the network inputs directly into the output layer unit
                IEnumerable<float> result = Enumerable.Range(0, ny).Select(
                    hy =>
                    {
                        int sk = hy * (1 + nx);
                        var oy = weights.Skip(sk).First();  // the bias
                        oy += values.Zip(weights.Skip(sk + 1), (x, w) => w * x).Sum();
                        return unaryop(oy);
                    });
                return result;
            }
        }
    }
}
