#ifndef _GAMBOGE_NNET_H
#define _GAMBOGE_NNET_H 1

#include <functional>
#include <iterator>
#include <numeric>
#include <cmath>

namespace gamboge
{

	//! @brief  logistic transfer functor
	//!
	template< class T >
	struct logistic_output : public std::unary_function<T,T>
	{
		T operator()( const T& x ) const
		{
			return static_cast<T>( 1 ) / ( static_cast<T>( 1 ) + std::exp( -x ) );
		}
	};

	//! @brief  unity transfer functor
	//!
	template< class T >
	struct linear_output : public std::unary_function<T,T>
	{
		T operator()( const T& x ) const
		{
			return x;
		}
	};

	template< typename FwdIter1, typename FwdIter2, typename OutIter, typename _Size, typename UnaryOp >
	OutIter
	_annet( FwdIter1 rbegin, FwdIter2 wtbegin, OutIter result, _Size n, _Size m, _Size k, UnaryOp outtransform )
	{
		FwdIter2 itw = wtbegin;
		typedef typename std::iterator_traits<OutIter>::value_type VT;


		if ( m > 0 )
		{
			// compute each of the hidden layer outputs
			VT hidden_out[ m ];

			for ( int hk = 0; hk < m; ++hk )
			{
				VT bias = *itw;
				++itw;
				FwdIter2 itw_end = itw + n;
				VT oh = std::inner_product( itw, itw_end, rbegin, bias );
				itw = itw_end;
				hidden_out[ hk ] = outtransform( oh );
			}

			// feed the hidden layer outputs into the output layer units
			for ( int ok = 0; ok < k; ++ok )
			{
				VT bias = *itw;
				++itw;
				FwdIter2 itw_end = itw + m;
				VT oh = std::inner_product( itw, itw_end, hidden_out, bias );
				itw = itw_end;
				*result++ = outtransform( oh );
			}
		}
		else  // m is 0
		{
			// feed the network inputs directly into the output layer units
			for ( int ok = 0; ok < k; ++ok )
			{
				VT bias = *itw;
				++itw;
				FwdIter2 itw_end = itw + n;
				VT oh = std::inner_product( itw, itw + n, rbegin, bias );
				itw = itw_end;
				*result++ = outtransform( oh );
			}
		}
		return result;
	}

	template< typename FwdIter1, typename FwdIter2, typename OutIter, typename _Size >
	OutIter
	annet( FwdIter1 rbegin, FwdIter2 wtbegin, OutIter result, _Size n, _Size m, _Size k )
	{
		typedef typename std::iterator_traits<OutIter>::value_type VT;
		logistic_output< VT > outtransform;
		return _annet( rbegin, wtbegin, result, n, m, k, outtransform );
	}

	//! @brief  Compute artificial neural network output.
	//!
	//! Computes output of an artificial neural network with @a n inputs,
	//! @a m hidden layer units and @a k output unit.
	//! Network input values are read from the range [ rbegin, rbegin + n ).
	//!
	//! For each unit the inner product of its inputs and weights is added to
	//! a bias weight. This result is then fed into an output transfer function
	//! often a logistic function, to produce the unit's output.
	//!    y = L( bias + &lang; x, w &rang; )
	//!
	//! Each hidden layer unit has one bias input and @a n inputs connected to the @a n
	//! network inputs. For each of the unit's inputs there is a corresponding weight. The
	//! first @f$ 1 + n @f$ values starting at @a wtbegin are the first hidden layer
	//! unit's bias and input weights; the first value is the bias and the remaining @a n
	//! weights apply to the @a n network inputs in order. The next 1 + n values
	//! in the weight range are the second hidden layer unit's weights, etc.
	//!
	//! When @a m is greater than 0, the output layer units each have one bias input and
	//! @a m inputs connected to the @a m hidden layer outputs. The output layer
	//! unit has 1+m weights and these are the last values in the range
	//! [ wtbegin, wtbegin + (n+2)*m+1 ).
	//!
	//! If @a m equals 0 then there is no hidden layer and the network inputs
	//! connect directly to the output units. In this case the values in
	//! the range [ wtbegin, wtbegin + n+1 ) apply to the output unit;
	//! the first value is the bias and the remaining @a n weights apply
	//! to the @a n network inputs in order.
	//!
	//! @param rbegin   start of input value range
	//! @param wtbegin  start of weights input range
	//! @param result   start of output range
	//! @param n        count of neural network inputs
	//! @param m        count of hidden layer units
	//! @param k        count of neural network outputs
	//! @param outtransform  output transform used for each unit
	//! @return         iterator pointing just after values written to result
	//!
	template< typename FwdIter1, typename FwdIter2, typename OutIter, typename _Size, typename UnaryOp >
	OutIter
	annet( FwdIter1 rbegin, FwdIter2 wtbegin, OutIter result, _Size n, _Size m, _Size k, UnaryOp outtransform )
	{
		return _annet( rbegin, wtbegin, result, n, m, k, outtransform );
	}
}

#endif
