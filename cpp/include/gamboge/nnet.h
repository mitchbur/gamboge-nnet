#ifndef _GAMBOGE_NNET_H
#define _GAMBOGE_NNET_H 1

#include <functional>
#include <iterator>
#include <numeric>
#include <cmath>

namespace gamboge
{

	template< class T >
	struct logistic_output : public std::unary_function<T,T>
	{
		T operator()( const T& x ) const
		{
			return static_cast<T>( 1 ) / ( static_cast<T>( 1 ) + std::exp( -x ) );
		}
	};

	template< class T >
	struct linear_output : public std::unary_function<T,T>
	{
		T operator()( const T& x ) const
		{
			return x;
		}
	};

	//! @brief  Compute artificial neural network output.
	//!
	//! Computes output of an artificial neural network with @a n inputs,
	//! @a m hidden layer units and one output unit.
	//! Network input values are read from the range @f$ [ rbegin, rbegin + n ) @f$.
	//!
	//! For each unit the inner product of its inputs and weights is added to
	//! a bias weight. This result is then fed into a logistic function to
	//! produce the unit's output.
	//! @f$ y = L( bias + &lang; x, w &rang; ) @f$
	//!
	//! Each hidden layer unit has one bias input and @a n inputs connected to the @a n
	//! network inputs. For each of the unit's inputs there is a corresponding weight. The
	//! first @f$ 1 + n @f$ values starting at @a wtbegin are the first hidden layer
	//! unit's bias and input weights; the first value is the bias and the remaining @a n
	//! weights apply to the @a n network inputs in order. The next @f$ 1 + n @f$ values
	//! in the weight range are the second hidden layer unit's weights, etc.
	//!
	//! When @a m is greater than 0, the one output layer unit has one bias input and
	//! @a m inputs connected to the @a m hidden layer outputs. The output layer
	//! unit has 1+m weights and these are the last values in the range
	//! @f$ [ wtbegin, wtbegin + (n+2)*m+1 ) @f$.
	//!
	//! If @a m equals 0 then there is no hidden layer and the network inputs
	//! connect directly to the one output unit. In this case the values in
	//! the range @f$ [ wtbegin, wtbegin + n+1 ) @f$ apply to the output unit;
	//! the first value is the bias and the remaining @a n weights apply
	//! to the @a n network inputs in order.
	//!
	//! @param rbegin   start of input value range
	//! @param wtbegin  start of weights input range
	//! @param n        count of neural network inputs
	//! @param m        count of hidden layer units
	//! @return         neural network output value
	//!
	template< typename FwdIter1, typename FwdIter2, typename _Size >
	typename std::iterator_traits< FwdIter1 >::value_type
	annet( FwdIter1 rbegin, FwdIter2 wtbegin, _Size n, _Size m )
	{
		FwdIter2 itw = wtbegin;
		typedef typename std::iterator_traits<FwdIter1>::value_type VT;
		logistic_output<VT> outtransform;


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

			// feed the hidden layer outputs into the output layer unit
			{
				VT bias = *itw;
				++itw;
				VT oh = std::inner_product( itw, itw + m, hidden_out, bias );
				return outtransform( oh );
			}
		}
		else  // m is 0
		{
			// feed the network inputs directly into the output layer unit
			VT bias = *itw;
			++itw;
			VT oh = std::inner_product( itw, itw + n, rbegin, bias );
			return outtransform( oh );
		}
	}

	//! @brief  Compute artificial neural network output.
	//!
	//! Computes output of an artificial neural network with @a n inputs,
	//! @a m hidden layer units and one output unit. @a outtransform is the
	//! output transfer function for hidden units and the output unit.
	//!
	//! @param rbegin   start of input value range
	//! @param wtbegin  start of weights input range
	//! @param n        length of network input sequence
	//! @param m        number of hidden layer units
	//! @param outtransform output transfer function for hidden and output units
	//! @return         neural network output value
	//!
	template< typename FwdIter1, typename FwdIter2, typename _Size, typename UnaryOp >
	typename std::iterator_traits<FwdIter1>::value_type
	annet( FwdIter1 rbegin, FwdIter2 wtbegin, _Size n, _Size m, UnaryOp outtransform )
	{
		FwdIter2 itw = wtbegin;
		typedef typename std::iterator_traits<FwdIter1>::value_type VT;


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

			// feed the hidden layer outputs into the output layer unit
			{
				VT bias = *itw;
				++itw;
				VT oh = std::inner_product( itw, itw + m, hidden_out.begin(), bias );
				return outtransform( oh );
			}
		}
		else  // m is 0
		{
			// feed the network inputs directly into the output layer unit
			VT bias = *itw;
			++itw;
			VT oh = std::inner_product( itw, itw + n, rbegin, bias );
			return outtransform( oh );
		}
	}

}

#endif
