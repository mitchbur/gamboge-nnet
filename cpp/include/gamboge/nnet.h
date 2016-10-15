//! @file gamboge/nnet.h
//! gamboge neural network library header file

#ifndef _GAMBOGE_NNET_H
#define _GAMBOGE_NNET_H 1

#include <functional>
#include <iterator>
#include <numeric>
#include <cmath>

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

	//! @cond
	//! implementation, target template code for functional dispatch
	template< typename InIter1, typename InIter2, typename OutIter, typename Size, typename UnaryOp >
	OutIter
	_neural_network( InIter1 rbegin, InIter2 wtbegin, OutIter result,
		Size nx, Size nh, Size ny, UnaryOp unaryop )
	{
		InIter2 itw = wtbegin;
		typedef typename std::iterator_traits<OutIter>::value_type VT;


		if ( nh > 0 )
		{
			// compute each of the hidden layer outputs
			VT hidden_out[ nh ];

			for ( int hk = 0; hk < nh; ++hk )
			{
				VT bias = *itw;
				++itw;
				InIter2 itw_end = itw + nx;
				VT oh = std::inner_product( itw, itw_end, rbegin, bias );
				itw = itw_end;
				hidden_out[ hk ] = unaryop( oh );
			}

			// feed the hidden layer outputs into the output layer units
			for ( int ok = 0; ok < ny; ++ok )
			{
				VT bias = *itw;
				++itw;
				InIter2 itw_end = itw + nh;
				VT oh = std::inner_product( itw, itw_end, hidden_out, bias );
				itw = itw_end;
				*result++ = unaryop( oh );
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
				VT oh = std::inner_product( itw, itw_end, rbegin, bias );
				itw = itw_end;
				*result++ = unaryop( oh );
			}
		}
		return result;
	}
	//! @endcond

	//! Compute artificial neural network outputs
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
	//! hidden-layer and output-layer unit in the network.
	//!
	//! @par a neural unit output is:
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
	neural_network( InIter1 firstx, InIter2 firstw, OutIter result,
		Size nx, Size nh, Size ny )
	{
		typedef typename std::iterator_traits<OutIter>::value_type VT;
		return _neural_network( firstx, firstw, result, nx, nh, ny, logistic_output< VT >() );
	}

	//! Compute artificial neural network outputs
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
	//! @p unaryop is applied at the output of each hidden-layer and output-layer
	//! unit in the network.
	//!
	//! @par a neural unit output is:
	//!    @a y = @b U( @a bias + &lang; @a x, @a w &rang; ) @n
	//! @p unaryop (@b U) applied to the sum of the unit's bias and
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
	neural_network( InIter1 firstx, InIter2 firstw, OutIter result,
		Size nx, Size nh, Size ny, UnaryOp unaryop )
	{
		return _neural_network( firstx, firstw, result, nx, nh, ny, unaryop );
	}
}

#endif
