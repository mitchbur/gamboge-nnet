library( nnet )

# train a neural network for the iris dataset
# @param seed     random seed
# @param formula  nnet formula parameter
# @param p        proportion of data used for learning, (0 , 1]
# @param npred    number of predictions to return
# @param ...      other parameters passed to nnet
# @returns  list: wts= neural network weights (nnet$wts),
#                 vdata= verification input data,
#                 pred= verification output data,
#                 nnet= trained ANN
iris.nnet <- function( seed=NULL, formula=Species ~ ., p=0.5, npred=10, ... )
{
	set.seed( seed )

	# create random subset of iris data frame rows
	N <- dim( iris )[1]
	M <- dim( iris )[2]
	iris.train <- iris[ sample( N, as.integer( p * N ) ), ]

	# train the neural network using the subset of iris data
	nn <- nnet( formula=formula, data=iris.train, ... )

	# predictions
	# random selection of iris data frame rows, only the nnet term columns
	iris.v <- iris[ sample( N, npred ), attr( nn$terms, "term.labels" ) ]
	pred <- predict( nn, newdata=iris.v )

	return( list( wts=nn$wts, vdata=iris.v, pred=pred, nnet=nn ) )
}

floats.as.string <- function( v )
{
	if ( is.vector(v) )
	{
		vn <- v[ sapply( v, is.numeric ) ]
		C.float.literals <- paste( signif( vn, 8 ), "F", sep="" )
		vstr <- paste( C.float.literals, collapse=", " )
	}
	else
	{
		rv <- apply( v, 1, floats.as.string )
		vstr <- paste( "\n\t", paste( rv, collapse=",\n\t" ), "\n" )
	}
	return( vstr )
}

cat.iris.nnet <- function( L )
{
	cat( paste( "topology =\n{", paste( L$nnet$n, collapse=", " ), "};\n" ))
	cat( paste( "weights =\n{", floats.as.string( L$wts ) ), "};\n" )
	cat( paste( "verif_data =\n{", floats.as.string( L$vdata[ , sapply(L$vdata,is.numeric) ] ) ), "};\n" )
	cat( paste( "predicted =\n{", floats.as.string( L$pred ) ), "};\n" )
}

testdata_423 <- function( )
{
	L <- iris.nnet( seed=678, npred=20, size=2, decay=0.01 )
	cat.iris.nnet( L )
}

testdata_321 <- function( )
{
	L <- iris.nnet( seed=627, npred=20,
		 	I( Species == 'versicolor' ) ~ Petal.Width + Sepal.Length + Petal.Length,
			size=2, decay=0.007, maxit=300 )
	cat.iris.nnet( L )
}
