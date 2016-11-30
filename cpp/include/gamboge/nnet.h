//! @file gamboge/nnet.h
//! gamboge neural network library header file

#ifndef _GAMBOGE_NNET_H
#define _GAMBOGE_NNET_H 1

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <cmath>
#include <new>

namespace gamboge
{
	//! linear transfer functor
	//!
	//! @par the identity function:
	//! y = x
	//!
	template< class T >
	struct linear_output : public std::unary_function<T,T>
	{
		T operator()( const T& x ) const
		{
			return x;
		}
	};

	//! logistic transfer functor
	//!
	//! @par the logistic function:
	//! @f$ y = \frac{1}{1 + e^{-x}} @f$
	//!
	template< class T >
	struct logistic_output : public std::unary_function<T,T>
	{
		T operator()( const T& x ) const
		{
			return static_cast<T>( 1 ) / ( static_cast<T>( 1 ) + std::exp( -x ) );
		}
	};

	template< typename T, typename T2 >
	struct normexp_op : public std::unary_function< T, T2 >
	{
		normexp_op( T norm )
			: norm( norm )
		{}

		T2 operator( )( const T& x ) const
		{
			return std::exp( x - norm );
		}

		T norm;
	};

	template< typename FwdIter, typename FwdIter2 >
	FwdIter2
	_softmax( FwdIter first, FwdIter last, FwdIter2 result )
	{
		typedef typename std::iterator_traits<FwdIter>::value_type VT;
		typedef typename std::iterator_traits<FwdIter2>::value_type VT2;

		// determine maximum input value mx, then compute exp( x - mx )
		// for each input
		FwdIter maxin_p = std::max_element( first, last );
		FwdIter2 result_last = std::transform( first, last, result,
			normexp_op< VT, VT2 >(*maxin_p) );

		VT denom = std::accumulate( first, last, static_cast<VT>(0) );

		std::transform( result, result_last, result, std::bind2nd( std::divides< VT2 >(), denom ) );
		return result_last;
	}

	//! @cond
	//! implementation, target template code for functional dispatch
	template< typename InIter1, typename InIter2, typename OutIter, typename Size, typename UnaryOp >
	OutIter
	_evaluate_neural_network( InIter1 rbegin, InIter2 wtbegin, OutIter result,
		Size nx, Size nh, Size ny, UnaryOp unaryop )
	{
		InIter2 itw = wtbegin;
		typedef typename std::iterator_traits<OutIter>::value_type VT;
		// create a buffer for input values, hidden-layer unit outputs and
		// output-layer unit linear outputs
		VT* unitsbuf = new (std::nothrow) VT[ nx + nh + ny ];
		if ( unitsbuf == 0 )
		{
			return result;
		}
		VT* inbuf = &(unitsbuf[0]);
		VT* hidden_out = &(unitsbuf[nx]);
		VT* linout = &(unitsbuf[nx+nh]);

		// to support Input Iterator type
		// read using rbegin in one forward pass
		// data is buffered because it will be read many times
		for ( int k = 0; k < nx; ++k )
		{
			inbuf[k] = *rbegin++;
		}

		if ( nh > 0 )
		{
			// compute each of the hidden layer outputs
			for ( int hk = 0; hk < nh; ++hk )
			{
				VT bias = *itw;
				++itw;
				InIter2 itw_end = itw + nx;
				VT oh = std::inner_product( itw, itw_end, inbuf, bias );
				itw = itw_end;
				hidden_out[ hk ] = logistic_output< VT >()( oh );
			}

			// feed the hidden layer outputs into the output layer units
			for ( int ok = 0; ok < ny; ++ok )
			{
				VT bias = *itw;
				++itw;
				InIter2 itw_end = itw + nh;
				linout[ok] = std::inner_product( itw, itw_end, &(hidden_out[0]), bias );
				itw = itw_end;
			}
		}
		else  // nh is 0
		{
			// feed the network inputs directly into the output layer units
			for ( int ok = 0; ok < ny; ++ok )
			{
				VT bias = *itw;
				++itw;
				InIter2 itw_end = itw + nx;
				linout[ok] = std::inner_product( itw, itw_end, inbuf, bias );
				itw = itw_end;
			}
		}

		if ( ny > 1 )
		{
			_softmax( linout, &(linout[ny]), linout );
			std::copy( linout, &(linout[ny]), result );
		}
		else
		{
			std::transform( linout, &(linout[ny]), result, unaryop );
		}

		delete[] unitsbuf;

		return result;
	}
	//! @endcond

	//! Evaluate artificial neural network outputs
	//!
	//! @param firstx   start of input value range
	//! @param firstw   start of weights input range
	//! @param result   start of output range
	//! @param nx       input count
	//! @param nh       hidden-layer count
	//! @param ny       output count
	//! @return iterator marking end of result sequence
	//!
	//! Computes feed-forward artificial neural network outputs. The network
	//! has @p nx inputs, @p nh hidden-layer units and @p ny output units.
	//! Network input values are read from the range
	//! [ @p firstx, @p firstx + @p nx ).
	//! Network output values are written to the range
	//! [ @p result, @p result + @p ny ).
	//! Bias and input weights for the hidden-layer and output-layer units
	//! are read from the range [ @p firstw, @p firstw + @a v ), where the
	//! number of weights @a v varies based upon @p nx, @p nh and @p ny.
	//! @p the logistics_output function is applied at the output of each
	//! hidden-layer. If there is a single output-layer unit then the
	/// logistics_output function is applied at the output of  the output-layer
	/// unit. If there are multiple output-layer units then the softmax operator
	/// is applied to the output-layer units' linear outputs and the result is
	/// the output for the the network.
	//!
	//! @par a hidden-layer neural unit output is:
	//!    @a y = @b L( @a bias + &lang; @a x, @a w &rang; ) @n
	//! The logistics operator @b L applied to the sum of the unit's bias and
	//! the inner product of the unit's inputs and weights.
	//!
	//! <em>If @p nh is greater than 0</em>, the network inputs feed the
	//! hidden-layer units and the hidden-layer units feed the output-layer units.
	//! In this case the weights sequence starts
	//! with blocks for each of the hidden-layer units followed by blocks for
	//! each of the output-layer units.
	//! Each hidden-layer block consists of 1 + @p nx values; the first value
	//! is the bias for the unit and the remainder are weights for each of the
	//! network inputs.
	//! Each output-layer block consists of 1 + @p nh values; the first value
	//! is the bias for the unit and the remainder are weights for each of the
	//! hidden-layer unit outputs.
	//! The number of weights is: @n
	//! @a v = @p nh * ( 1 + @p nx ) + @p ny * ( 1 + @p nh ).
	//!
	//! <em>If @a nh equals 0</em>, there are no hidden-layer units and the network
	//! inputs directly feed the output-layer units.
	//! In this case the weights sequence consists of blocks for the output-layer
	//! units.
	//! Each output-layer block consists of 1 + @p nx values; the first value
	//! is the bias for the unit and the remainder are weights for each of the
	//! network inputs.
	//! The number of weights is: @n
	//! @a v = @p ny * ( 1 + @p nx ).
	//!
	template< typename InIter1, typename InIter2, typename OutIter, typename Size >
	OutIter
	evaluate_neural_network( InIter1 firstx, InIter2 firstw, OutIter result,
		Size nx, Size nh, Size ny )
	{
		typedef typename std::iterator_traits<OutIter>::value_type VT;
		return _evaluate_neural_network( firstx, firstw, result, nx, nh, ny, logistic_output< VT >() );
	}

	//! Evaluate artificial neural network outputs
	//!
	//! @param firstx   start of input value sequence
	//! @param firstw   start of weights sequence
	//! @param result   start of output sequence
	//! @param nx       input count
	//! @param nh       hidden-layer count
	//! @param ny       output count
	//! @param unaryop  output transform for units
	//! @return iterator marking end of result sequence
	//!
	//! Computes feed-forward artificial neural network outputs. The network
	//! has @p nx inputs, @p nh hidden-layer units and @p ny output units.
	//! Network input values are read from the range
	//! [ @p firstx, @p firstx + @p nx ).
	//! Network output values are written to the range
	//! [ @p result, @p result + @p ny ).
	//! Bias and input weights for the hidden-layer and output-layer units
	//! are read from the range [ @p firstw, @p firstw + @a v ), where the
	//! number of weights @a v varies based upon @p nx, @p nh and @p ny.
	//! @p the logistics_output function is applied at the output of each
	//! hidden-layer unit. If there is a single output-layer unit then
	/// @p unaryop is applied at the output of  the output-layer
	/// unit. If there are multiple output-layer units then the softmax operator
	/// is applied to the output-layer units' linear outputs and the result is
	/// the output for the the network.
	//!
	//! @par a hidden-layer neural unit output is:
	//!    @a y = @b L( @a bias + &lang; @a x, @a w &rang; ) @n
	//! The logistics operator @b L applied to the sum of the unit's bias and
	//! the inner product of the unit's inputs and weights.
	//!
	//! <em>If @p nh is greater than 0</em>, the network inputs feed the
	//! hidden-layer units and the hidden-layer units feed the output-layer units.
	//! In this case the weights sequence starts
	//! with blocks for each of the hidden-layer units followed by blocks for
	//! each of the output-layer units.
	//! Each hidden-layer block consists of 1 + @p nx values; the first value
	//! is the bias for the unit and the remainder are weights for each of the
	//! network inputs.
	//! Each output-layer block consists of 1 + @p nh values; the first value
	//! is the bias for the unit and the remainder are weights for each of the
	//! hidden-layer unit outputs.
	//! The number of weights is: @n
	//! @a v = @p nh * ( 1 + @p nx ) + @p ny * ( 1 + @p nh ).
	//!
	//! <em>If @a nh equals 0</em>, there are no hidden-layer units and the network
	//! inputs directly feed the output-layer units.
	//! In this case the weights sequence consists of blocks for the output-layer
	//! units.
	//! Each output-layer block consists of 1 + @p nx values; the first value
	//! is the bias for the unit and the remainder are weights for each of the
	//! network inputs.
	//! The number of weights is: @n
	//! @a v = @p ny * ( 1 + @p nx ).
	//!
	template< typename InIter1, typename InIter2, typename OutIter, typename Size, typename UnaryOp >
	OutIter
	evaluate_neural_network( InIter1 firstx, InIter2 firstw, OutIter result,
		Size nx, Size nh, Size ny, UnaryOp unaryop )
	{
		return _evaluate_neural_network( firstx, firstw, result, nx, nh, ny, unaryop );
	}

	//! artificial neural network
	template < typename InIter, typename InIterWt, typename OutIter, typename Size >
	class neural_network
	{
	public:
		// typedef typename std::iterator_traits<OutIter>::value_type value_type;

		//! constructor, artificial neural network
		//!
		neural_network( int n, int m, int k, InIterWt wts )
		: input_count( n ),
		  hidden_count( m ),
		  output_count( k ),
		  weights( wts )
		{
		}

		//! Evaluate artificial neural network outputs
		//!
		//! @param result   start of output sequence
		//! @param values   start of input value sequence
		//! @return iterator marking end of result sequence
		//!
		//! Example
		//! @code
		//! {
		//! 	typedef gamboge::neural_network< const double*, const double*, double*, unsigned > nnet_type;
		//! 	const unsigned in_count = 3;
		//! 	const unsigned hidden_count = 2;
		//! 	const unsigned out_count = 1;
		//! 	const double wts[ ] = {
		//! 		 0.56974212, -1.5468268,  1.494846, -2.8907045,
		//! 		-6.5020564,   3.0203401, -1.7088961, 2.5260361,
		//! 		 3.393649,   -6.7710899, -7.2983476
		//! 		};
		//! 	nnet_type example_nnet( in_count, hidden_count, out_count, &(wts[0]) );
		//! 	const double nn_in[ in_count ] = { 1.4, 6.8, 4.8 };
		//! 	double nn_out[ out_count ];
		//!
		//! 	example_nnet.evaluate( nn_out, nn_in );
		//! }
		//! @endcode
		OutIter evaluate( OutIter result, InIter values ) const
		{
			return evaluate_neural_network( values, weights, result,
				input_count, hidden_count, output_count );
		}

	private:
		Size input_count;
		Size hidden_count;
		Size output_count;
		InIterWt weights;
	};
}

#endif
