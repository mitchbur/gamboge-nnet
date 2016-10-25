#include "gamboge/nnet.h"
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "cppunit/TestCase.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestResult.h"
#include <cppunit/ui/text/TestRunner.h>

template< typename FP >
struct absdiff : public std::binary_function< FP, FP, FP >
{
	FP operator()( const FP& a, const FP& b ) const
	{
		return std::abs( a - b );
	}
};

template <typename FP>
class gamboge_nnet_tester
{
public:
	gamboge_nnet_tester(
		unsigned in_count, unsigned hidden_count, unsigned out_count, const FP* wts,
		unsigned verif_count, const FP* verif_in, const FP* expected_out )
	 : in_count( in_count ),
	 hidden_count( hidden_count ),
	 out_count( out_count ),
	 wts( wts ),
	 verif_count( verif_count ),
	 verif_in( verif_in ),
	 expected_out( expected_out )
	{ }

	void run_test( )
	{
		FP max_error = static_cast<FP>( 0 );
		std::vector<FP> nn_out( out_count );

		for ( unsigned k = 0; k < verif_count; ++k )
		{
			gamboge::neural_network( &(verif_in[k*in_count]), wts, nn_out.begin(),
				in_count, hidden_count, out_count );

			// use "inner_product" to determine maximum absolute difference
			// of result and expected values; the absdiff operator will be
			// called for each corresponding expected and result values
			// and the max operator will select the greatest value seen
			max_error = std::inner_product( nn_out.begin(), nn_out.end(),
				&(expected_out[k*out_count]), max_error,
				fmax, absdiff<FP>() );
		}

		CPPUNIT_ASSERT_ASSERTION_PASS_MESSAGE( "check maximum absolute error",
			CPPUNIT_ASSERT_LESS( 7.8E-7F, max_error ) );
	}

private:
	unsigned in_count;
	unsigned hidden_count;
	unsigned out_count;
	const FP* wts;
	unsigned verif_count;
	const FP* verif_in;
	const FP* expected_out;
};

// test neural network 6-3-1 topology
class ann631TestCase : public CppUnit::TestCase
{
public:
	ann631TestCase( std::string name )
	: CppUnit::TestCase( name )
	{}

	void runTest( )
	{
		gamboge_nnet_tester<float> tester( 6U, 3U, 1U, test_wts,
			 8U, test_inputs, expected_outputs );
		tester.run_test( );
	}

private:
	static const float test_wts[];
	static const float test_inputs[];
	static const float expected_outputs[];
};

const float ann631TestCase::test_wts[] = {
	// b->h1          i1->h1         i2->h1         i3->h1         i4->h1         i5->h1         i6->h1
	  6.31414733F,   0.65097616F,   9.57090502F,   0.09918807F,   0.34747524F,  -0.22119165F,  -1.46227569F,
	// b->h2          i1->h2         i2->h2         i3->h2         i4->h2         i5->h2         i6->h2
	 -2.90137623F,   5.28471412F, -18.85611073F,  -1.23064304F,   0.67967101F,  -0.52377262F,   2.18077394F,
	// b->h3          i1->h3         i2->h3         i3->h3         i4->h3         i5->h3         i6->h3
	  2.73558004F,   1.84685605F,  -1.34649983F,   9.83496163F,  -8.08858473F,   1.23608243F, -11.48135362F,
	// b->o           h1->o          h2->o          h3->o
	 -3.51048773F,  -7.08398606F,  11.45778956F, -19.95901352F
};

const float ann631TestCase::test_inputs[] = {
	 0.37182781F, -0.8311404F,  0.4259828F, -1.4220337F,  0.18336578F, -2.2287368F,
	 0.54980689F, -0.8311404F,  1.0703898F, -0.5883251F,  0.18336578F, -1.0588553F,
	-1.06745579F,  1.5445350F, -0.7493098F,  1.1007028F, -1.06960449F,  0.6201034F,
	 0.09560690F, -0.6512472F, -0.7493098F,  0.3154891F,  0.08288661F,  0.6201034F,
	-2.43320564F,  2.5265420F, -0.4770242F,  1.2647570F, -1.09247945F,  0.5813455F,
	-1.81966033F,  1.3682108F, -0.4770242F,  1.2093380F, -1.01113866F,  0.5813455F,
	-0.67703554F, -0.7889860F, -0.4770242F,  0.4291913F,  0.13391541F,  0.5813455F,
	-0.39566591F, -0.7031708F, -0.7544150F,  0.2704232F,  0.04057550F,  0.6718149F
};

const float ann631TestCase::expected_outputs[] = {
	0.00000e+00F,
	0.00000e+00F,
	2.50535e-05F,
	9.96561e-01F,
	2.50535e-05F,
	2.50535e-05F,
	9.99381e-01F,
	9.98889e-01F
};

//
// test neural network 3-2-1 topology
class ann321TestCase : public CppUnit::TestCase
{
public:
	ann321TestCase( std::string name )
	: CppUnit::TestCase( name )
	{}

	void runTest( )
	{
		gamboge_nnet_tester<float> tester( 3U, 2U, 1U, weights,
			20U, verif_data, predicted );
		tester.run_test( );
	}

private:
	static const float weights[];
	static const float verif_data[];
	static const float predicted[];
};
const float ann321TestCase::weights[] =
{
	0.56974212F, -1.5468268F, 1.494846F, -2.8907045F,
	-6.5020564F, 3.0203401F, -1.7088961F, 2.5260361F,
	3.393649F, -6.7710899F, -7.2983476F
};
const float ann321TestCase::verif_data[] =
{
	1.4F, 6.8F, 4.8F,
	2.3F, 6.4F, 5.3F,
	1.3F, 5.7F, 4.1F,
	0.2F, 4.7F, 1.3F,
	1.4F, 7.0F, 4.7F,
	2.5F, 6.7F, 5.7F,
	1.9F, 5.8F, 5.1F,
	0.2F, 4.8F, 1.6F,
	0.1F, 4.3F, 1.1F,
	1.5F, 6.0F, 5.0F,
	1.3F, 5.7F, 4.2F,
	1.3F, 5.5F, 4.0F,
	1.3F, 5.6F, 4.1F,
	2.2F, 7.7F, 6.7F,
	0.2F, 5.4F, 1.7F,
	1.8F, 7.3F, 6.3F,
	0.3F, 5.7F, 1.7F,
	0.2F, 5.1F, 1.6F,
	1.0F, 5.0F, 3.5F,
	1.4F, 6.1F, 4.7F
};
const float ann321TestCase::predicted[] =
{
	0.90864414F ,
	0.028647561F,
	0.91948747F ,
	0.039752923F,
	0.93738283F ,
	0.022455461F,
	0.039801861F,
	0.047723386F,
	0.03885664F ,
	0.27093682F ,
	0.90293789F ,
	0.91397009F ,
	0.90881932F ,
	0.022725684F,
	0.040591468F,
	0.036147026F,
	0.038563865F,
	0.042004902F,
	0.94917055F ,
	0.76039408F
};

// test neural network 4-2-3 topology
class ann423TestCase : public CppUnit::TestCase
{
public:
	ann423TestCase( std::string name )
	: CppUnit::TestCase( name )
	{}

	void runTest( )
	{
		gamboge_nnet_tester<float> tester( 4U, 2U, 3U, weights,
			20U, verif_data, predicted );
		tester.run_test( );
	}

private:
	static const float weights[];
	static const float verif_data[];
	static const float predicted[];
};
const float ann423TestCase::weights[] =
{
	 -7.5744544F, -0.98429384F, -1.216025F, 1.9840944F, 4.3170568F,
	 0.35806831F, 0.47724404F, 1.5541206F, -2.4603607F, -0.99349176F,
	 -1.6232478F, -2.1703089F, 6.0064449F,
	 3.9738482F, -5.5195306F, -5.2175259F,
	 -2.3506086F, 7.6898515F, -0.78892375F
};
const float ann423TestCase::verif_data[] =
{
	4.4F, 3.0F, 1.3F, 0.2F,
	5.1F, 3.8F, 1.9F, 0.4F,
	4.9F, 3.0F, 1.4F, 0.2F,
	5.4F, 3.4F, 1.5F, 0.4F,
	7.2F, 3.2F, 6.0F, 1.8F,
	5.6F, 2.7F, 4.2F, 1.3F,
	6.0F, 2.2F, 5.0F, 1.5F,
	4.8F, 3.1F, 1.6F, 0.2F,
	6.9F, 3.2F, 5.7F, 2.3F,
	5.0F, 3.5F, 1.6F, 0.6F,
	6.3F, 3.3F, 4.7F, 1.6F,
	5.5F, 2.3F, 4.0F, 1.3F,
	7.2F, 3.6F, 6.1F, 2.5F,
	5.7F, 2.6F, 3.5F, 1.0F,
	6.9F, 3.1F, 4.9F, 1.5F,
	6.2F, 2.8F, 4.8F, 1.8F,
	5.4F, 3.0F, 4.5F, 1.5F,
	4.6F, 3.4F, 1.4F, 0.3F,
	6.4F, 2.7F, 5.3F, 1.9F,
	4.6F, 3.2F, 1.4F, 0.2F
 };
const float ann423TestCase::predicted[] =
{
	0.99470267F, 0.0046660095F, 0.00063132111F,
	0.99456831F, 0.0047902576F, 0.0006414322F,
	0.99469299F, 0.0046749691F, 0.00063204182F,
	0.99512325F, 0.0042776915F, 0.00059906041F,
	0.0011873214F, 0.025462821F, 0.97334986F,
	0.0055198862F, 0.9890041F, 0.0054760163F,
	0.006172877F, 0.25299856F, 0.74082856F,
	0.99404932F, 0.0052711547F, 0.00067952264F,
	0.00017412947F, 0.0019279709F, 0.9978979F,
	0.994605F, 0.0047562526F, 0.00063874748F,
	0.0071978296F, 0.97461438F, 0.018187792F,
	0.0056746864F, 0.98769809F, 0.0066272266F,
	0.00013261192F, 0.0013410981F, 0.99852629F,
	0.0088674358F, 0.9884324F, 0.0027001664F,
	0.0059661661F, 0.98447395F, 0.0095598806F,
	0.0065476968F, 0.2689804F, 0.7244719F,
	0.0085873027F, 0.94774833F, 0.043664363F,
	0.99509328F, 0.0043053246F, 0.00060139864F,
	0.00059435524F, 0.0099872336F, 0.98941841F,
	0.99489201F, 0.0044910661F, 0.00061692014F
};

void
gamboge_nnet_runtests( )
{
	CppUnit::TextUi::TestRunner runner;
	CppUnit::TestSuite* suite = new CppUnit::TestSuite( "nnet test suite" );

	suite->addTest( new ann631TestCase( "nnet 6-3-1 topology" ) );
	suite->addTest( new ann321TestCase( "nnet 3-2-1 topology" ) );
	suite->addTest( new ann423TestCase( "nnet 4-2-3 topology" ) );

	runner.addTest( suite );
	runner.run( );
}
