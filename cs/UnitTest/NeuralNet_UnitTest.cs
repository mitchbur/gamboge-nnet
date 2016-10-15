using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Gamboge;

namespace GambogeUnitTest
{
    [TestClass]
    public class NeuralNet_UnitTest
    {
        static readonly float[] t631_test_wts = new float[] {
    		// b->h1          i1->h1         i2->h1         i3->h1         i4->h1         i5->h1         i6->h1
		  6.31414733F,   0.65097616F,   9.57090502F,   0.09918807F,   0.34747524F,  -0.22119165F,  -1.46227569F,
    		// b->h2          i1->h2         i2->h2         i3->h2         i4->h2         i5->h2         i6->h2
		 -2.90137623F,   5.28471412F, -18.85611073F,  -1.23064304F,   0.67967101F,  -0.52377262F,   2.18077394F,
    		// b->h3          i1->h3         i2->h3         i3->h3         i4->h3         i5->h3         i6->h3
		  2.73558004F,   1.84685605F,  -1.34649983F,   9.83496163F,  -8.08858473F,   1.23608243F, -11.48135362F,
    		// b->o           h1->o          h2->o          h3->o
		 -3.51048773F,  -7.08398606F,  11.45778956F, -19.95901352F
        };

        static readonly float[][] t631_test_inputs = new float[][] {
		    new float[] {  0.37182781F, -0.8311404F,  0.4259828F, -1.4220337F,  0.18336578F, -2.2287368F },
            new float[] {  0.54980689F, -0.8311404F,  1.0703898F, -0.5883251F,  0.18336578F, -1.0588553F },
            new float[] { -1.06745579F,  1.5445350F, -0.7493098F,  1.1007028F, -1.06960449F,  0.6201034F },
            new float[] {  0.09560690F, -0.6512472F, -0.7493098F,  0.3154891F,  0.08288661F,  0.6201034F },
            new float[] { -2.43320564F,  2.5265420F, -0.4770242F,  1.2647570F, -1.09247945F,  0.5813455F },
            new float[] { -1.81966033F,  1.3682108F, -0.4770242F,  1.2093380F, -1.01113866F,  0.5813455F },
            new float[] { -0.67703554F, -0.7889860F, -0.4770242F,  0.4291913F,  0.13391541F,  0.5813455F },
            new float[] { -0.39566591F, -0.7031708F, -0.7544150F,  0.2704232F,  0.04057550F,  0.6718149F }
	    };

	    static readonly float[] t631_expected_outputs = new float[] {
            0.00000e+00F,
            0.00000e+00F,
            2.50535e-05F,
            9.96561e-01F,
            2.50535e-05F,
            2.50535e-05F,
            9.99381e-01F,
            9.98889e-01F
        };

        [TestMethod]
        public void test_631()
        {
            NeuralNetwork t631_nnet = new NeuralNetwork(6, 3, t631_test_wts);
            for (int k = 0; k < 8; ++k)
            {
                float nn_out = t631_nnet.Compute( t631_test_inputs[k] );
                System.Console.Write("6-3-1 run {0:D2}:", k );
                System.Console.Write(" result={0:F6}", nn_out);
                System.Console.Write(", expected={0:F6}", t631_expected_outputs[k] );
                System.Console.WriteLine();
                Assert.IsTrue(Math.Abs(t631_expected_outputs[k] - nn_out) < 5E-7);
            }
        }
    }
}
