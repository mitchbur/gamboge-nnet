using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Gamboge;
using System.Collections.Generic;
using System.Linq;

namespace GambogeUnitTest
{
    [TestClass]
    public class NeuralNet_UnitTest_631
    {
        static int[] topology = { 6, 3, 1 };

        static readonly float[] test_wts = new float[] {
    		// b->h1          i1->h1         i2->h1         i3->h1         i4->h1         i5->h1         i6->h1
		  6.31414733F,   0.65097616F,   9.57090502F,   0.09918807F,   0.34747524F,  -0.22119165F,  -1.46227569F,
    		// b->h2          i1->h2         i2->h2         i3->h2         i4->h2         i5->h2         i6->h2
		 -2.90137623F,   5.28471412F, -18.85611073F,  -1.23064304F,   0.67967101F,  -0.52377262F,   2.18077394F,
    		// b->h3          i1->h3         i2->h3         i3->h3         i4->h3         i5->h3         i6->h3
		  2.73558004F,   1.84685605F,  -1.34649983F,   9.83496163F,  -8.08858473F,   1.23608243F, -11.48135362F,
    		// b->o           h1->o          h2->o          h3->o
		 -3.51048773F,  -7.08398606F,  11.45778956F, -19.95901352F
        };

        static readonly float[][] test_inputs = new float[][] {
            new float[] {  0.37182781F, -0.8311404F,  0.4259828F, -1.4220337F,  0.18336578F, -2.2287368F },
            new float[] {  0.54980689F, -0.8311404F,  1.0703898F, -0.5883251F,  0.18336578F, -1.0588553F },
            new float[] { -1.06745579F,  1.5445350F, -0.7493098F,  1.1007028F, -1.06960449F,  0.6201034F },
            new float[] {  0.09560690F, -0.6512472F, -0.7493098F,  0.3154891F,  0.08288661F,  0.6201034F },
            new float[] { -2.43320564F,  2.5265420F, -0.4770242F,  1.2647570F, -1.09247945F,  0.5813455F },
            new float[] { -1.81966033F,  1.3682108F, -0.4770242F,  1.2093380F, -1.01113866F,  0.5813455F },
            new float[] { -0.67703554F, -0.7889860F, -0.4770242F,  0.4291913F,  0.13391541F,  0.5813455F },
            new float[] { -0.39566591F, -0.7031708F, -0.7544150F,  0.2704232F,  0.04057550F,  0.6718149F }
        };

        static readonly float[][] expected_outputs = new float[][] {
            new float[] { 0.00000e+00F },
            new float[] { 0.00000e+00F },
            new float[] { 2.50535e-05F },
            new float[] { 9.96561e-01F },
            new float[] { 2.50535e-05F },
            new float[] { 2.50535e-05F },
            new float[] { 9.99381e-01F },
            new float[] { 9.98889e-01F }
        };

        [TestMethod]
        public void UnitTest_631()
        {
            NeuralNetwork test_nnet = new NeuralNetwork(
                topology[0], topology[1], topology[2], test_wts);
            for (int k = 0; k < expected_outputs.Length; ++k)
            {
                IEnumerable<float> nn_out = test_nnet.Compute(test_inputs[k]);
                double max_error = nn_out.Zip(expected_outputs[k],
                    (x, y) => { return Math.Abs(x - y); }).Max();

                Console.Write("{0}-{1}-{2} run {3:D2}:", topology[0], topology[1], topology[2], k);
                Console.Write(" maximum error={0:G6}", max_error);
                Console.WriteLine();
                Assert.IsTrue(max_error < 7.8E-7);
            }
        }
    }

    [TestClass]
    public class NeuralNet_UnitTest_321
    {
        static int[] topology = { 3, 2, 1 };

        static readonly float[] test_wts = new float[] {
             0.56974212F, -1.5468268F,  1.494846F, -2.8907045F,
            -6.5020564F,   3.0203401F, -1.7088961F, 2.5260361F,
             3.393649F,   -6.7710899F, -7.2983476F
        };

        static readonly float[][] test_inputs = new float[][] {
            new float[] { 1.4F, 6.8F, 4.8F },
            new float[] { 2.3F, 6.4F, 5.3F },
            new float[] { 1.3F, 5.7F, 4.1F },
            new float[] { 0.2F, 4.7F, 1.3F },
            new float[] { 1.4F, 7.0F, 4.7F },
            new float[] { 2.5F, 6.7F, 5.7F },
            new float[] { 1.9F, 5.8F, 5.1F },
            new float[] { 0.2F, 4.8F, 1.6F },
            new float[] { 0.1F, 4.3F, 1.1F },
            new float[] { 1.5F, 6.0F, 5.0F },
            new float[] { 1.3F, 5.7F, 4.2F },
            new float[] { 1.3F, 5.5F, 4.0F },
            new float[] { 1.3F, 5.6F, 4.1F },
            new float[] { 2.2F, 7.7F, 6.7F },
            new float[] { 0.2F, 5.4F, 1.7F },
            new float[] { 1.8F, 7.3F, 6.3F },
            new float[] { 0.3F, 5.7F, 1.7F },
            new float[] { 0.2F, 5.1F, 1.6F },
            new float[] { 1.0F, 5.0F, 3.5F },
            new float[] { 1.4F, 6.1F, 4.7F }
        };

        static readonly float[][] expected_outputs = new float[][] {
            new float[] { 0.90864414F  },
            new float[] { 0.028647561F },
            new float[] { 0.91948747F  },
            new float[] { 0.039752923F },
            new float[] { 0.93738283F  },
            new float[] { 0.022455461F },
            new float[] { 0.039801861F },
            new float[] { 0.047723386F },
            new float[] { 0.03885664F  },
            new float[] { 0.27093682F  },
            new float[] { 0.90293789F  },
            new float[] { 0.91397009F  },
            new float[] { 0.90881932F  },
            new float[] { 0.022725684F },
            new float[] { 0.040591468F },
            new float[] { 0.036147026F },
            new float[] { 0.038563865F },
            new float[] { 0.042004902F },
            new float[] { 0.94917055F  },
            new float[] { 0.76039408F  }
        };

        [TestMethod]
        public void UnitTest_321()
        {
            NeuralNetwork test_nnet = new NeuralNetwork(
                topology[0], topology[1], topology[2], test_wts);
            for (int k = 0; k < expected_outputs.Length; ++k)
            {
                IEnumerable<float> nn_out = test_nnet.Compute(test_inputs[k]);
                double max_error = nn_out.Zip(expected_outputs[k],
                    (x, y) => { return Math.Abs(x - y); }).Max();

                Console.Write("{0}-{1}-{2} run {3:D2}:", topology[0], topology[1], topology[2], k);
                Console.Write(" maximum error={0:G6}", max_error);
                Console.WriteLine();
                Assert.IsTrue(max_error < 7.8E-7);
            }
        }
    }

    [TestClass]
    public class NeuralNet_UnitTest_423
    {
        static int[] topology = { 4, 2, 3 };

        static readonly float[] test_wts = new float[] {
             -7.5744544F, -0.98429384F, -1.216025F,   1.9840944F,  4.3170568F,
              0.35806831F, 0.47724404F,  1.5541206F, -2.4603607F, -0.99349176F,
             -1.6232478F, -2.1703089F,   6.0064449F,
              3.9738482F, -5.5195306F,  -5.2175259F,
             -2.3506086F,  7.6898515F,  -0.78892375F
        };

        static readonly float[][] test_inputs = new float[][] {
            new float[] { 4.4F, 3.0F, 1.3F, 0.2F },
            new float[] { 5.1F, 3.8F, 1.9F, 0.4F },
            new float[] { 4.9F, 3.0F, 1.4F, 0.2F },
            new float[] { 5.4F, 3.4F, 1.5F, 0.4F },
            new float[] { 7.2F, 3.2F, 6.0F, 1.8F },
            new float[] { 5.6F, 2.7F, 4.2F, 1.3F },
            new float[] { 6.0F, 2.2F, 5.0F, 1.5F },
            new float[] { 4.8F, 3.1F, 1.6F, 0.2F },
            new float[] { 6.9F, 3.2F, 5.7F, 2.3F },
            new float[] { 5.0F, 3.5F, 1.6F, 0.6F },
            new float[] { 6.3F, 3.3F, 4.7F, 1.6F },
            new float[] { 5.5F, 2.3F, 4.0F, 1.3F },
            new float[] { 7.2F, 3.6F, 6.1F, 2.5F },
            new float[] { 5.7F, 2.6F, 3.5F, 1.0F },
            new float[] { 6.9F, 3.1F, 4.9F, 1.5F },
            new float[] { 6.2F, 2.8F, 4.8F, 1.8F },
            new float[] { 5.4F, 3.0F, 4.5F, 1.5F },
            new float[] { 4.6F, 3.4F, 1.4F, 0.3F },
            new float[] { 6.4F, 2.7F, 5.3F, 1.9F },
            new float[] { 4.6F, 3.2F, 1.4F, 0.2F }
        };

        static readonly float[][] expected_outputs = new float[][] {
            new float[] { 0.99470267F, 0.0046660095F, 0.00063132111F  },
            new float[] { 0.99456831F, 0.0047902576F, 0.0006414322F   },
            new float[] { 0.99469299F, 0.0046749691F, 0.00063204182F  },
            new float[] { 0.99512325F, 0.0042776915F, 0.00059906041F  },
            new float[] { 0.0011873214F, 0.025462821F, 0.97334986F    },
            new float[] { 0.0055198862F, 0.9890041F, 0.0054760163F    },
            new float[] { 0.006172877F, 0.25299856F, 0.74082856F      },
            new float[] { 0.99404932F, 0.0052711547F, 0.00067952264F  },
            new float[] { 0.00017412947F, 0.0019279709F, 0.9978979F   },
            new float[] { 0.994605F, 0.0047562526F, 0.00063874748F    },
            new float[] { 0.0071978296F, 0.97461438F, 0.018187792F    },
            new float[] { 0.0056746864F, 0.98769809F, 0.0066272266F   },
            new float[] { 0.00013261192F, 0.0013410981F, 0.99852629F  },
            new float[] { 0.0088674358F, 0.9884324F, 0.0027001664F    },
            new float[] { 0.0059661661F, 0.98447395F, 0.0095598806F   },
            new float[] { 0.0065476968F, 0.2689804F, 0.7244719F       },
            new float[] { 0.0085873027F, 0.94774833F, 0.043664363F    },
            new float[] { 0.99509328F, 0.0043053246F, 0.00060139864F  },
            new float[] { 0.00059435524F, 0.0099872336F, 0.98941841F  },
            new float[] { 0.99489201F, 0.0044910661F, 0.00061692014F  }
        };

        [TestMethod]
        public void UnitTest_423()
        {
            NeuralNetwork test_nnet = new NeuralNetwork(
                topology[0], topology[1], topology[2], test_wts );
            for (int k = 0; k < expected_outputs.Length; ++k)
            {
                IEnumerable<float> nn_out = test_nnet.Compute(test_inputs[k]);
                double max_error = nn_out.Zip(expected_outputs[k], 
                    (x, y) => { return Math.Abs(x - y); }).Max();

                Console.Write("{0}-{1}-{2} run {3:D2}:", topology[0], topology[1], topology[2], k);
                Console.Write(" maximum error={0:G6}", max_error);
                Console.WriteLine();
                Assert.IsTrue(max_error < 7.8E-7);
            }
        }
    }
}
