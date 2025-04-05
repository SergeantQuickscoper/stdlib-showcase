/**
* @license Apache-2.0
*
* Copyright (c) 2025 The Stdlib Authors.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

'use strict';

// MODULES //

var ndarray = require('@stdlib/ndarray');
var blas = require('@stdlib/blas');
var abs = require('@stdlib/math/base/special/abs');
var sqrt = require('@stdlib/math/base/special/sqrt');
var pow = require('@stdlib/math/base/special/pow');
var exp = require('@stdlib/math/base/special/exp');
var Float64Array = require('@stdlib/array/float64');

var array2d = ndarray.array;

/**
 * Creates a clone of an ndarray
 * 
 * @param {ndarray} arr - The array to clone
 * @returns {ndarray} A cloned copy of the array
 */
function cloneArray( arr ) {
    var shape = arr.shape;
    var n = shape[0];
    
    if ( shape.length === 1 ) {
        // 1D array
        var newData = new Float64Array( n );
        
        // Copy data
        for ( var i = 0; i < n; i++ ) {
            newData[i] = arr.get( i );
        }
        
        return array2d( newData, { 'shape': shape } );
    } else {
        // 2D array
        var m = shape[1];
        var newData = new Float64Array( n * m );
        var idx = 0;
        
        // Copy data
        for ( var i = 0; i < n; i++ ) {
            for ( var j = 0; j < m; j++ ) {
                newData[idx++] = arr.get( i, j );
            }
        }
        
        return array2d( newData, { 'shape': shape } );
    }
}

/**
* Computes the numerical gradient of a function at a point.
*
* @param {Function} f - The objective function f(x)
* @param {ndarray} x - Point at which to evaluate the gradient
* @param {number} h - Step size for finite difference (default: 1e-6)
* @returns {ndarray} The gradient vector
*/
function numericalGradient( f, x, h ) {
	h = h || 1e-6;
	var n = x.shape[0];
	var grad = array2d( new Float64Array( n ), {
		'shape': [n]
	} );
	
	// For each dimension
	var xPlus, xMinus;
	for ( var i = 0; i < n; i++ ) {
		// Create points for central difference
		xPlus = cloneArray( x );
		xMinus = cloneArray( x );
		
		// Adjust the ith coordinate
		xPlus.set( i, x.get( i ) + h );
		xMinus.set( i, x.get( i ) - h );
		
		// Compute central difference
		grad.set( i, ( f( xPlus ) - f( xMinus ) ) / ( 2 * h ) );
	}
	
	return grad;
}

/**
* Computes the numerical Hessian of a function at a point.
*
* @param {Function} f - The objective function f(x)
* @param {ndarray} x - Point at which to evaluate the Hessian
* @param {number} h - Step size for finite difference (default: 1e-6)
* @returns {ndarray} The Hessian matrix
*/
function numericalHessian( f, x, h ) {
	h = h || 1e-6;
	var n = x.shape[0];
	var hessian = array2d( new Float64Array( n * n ), {
		'shape': [n, n]
	} );
	
	// For each pair of dimensions
	var i, j;
	var xPlus, xMinus, xCenter;
	var xPlusPlus, xPlusMinus, xMinusPlus, xMinusMinus;
	
	for ( i = 0; i < n; i++ ) {
		for ( j = 0; j < n; j++ ) {
			if ( i === j ) {
				// Diagonal elements use central difference on f
				xPlus = cloneArray( x );
				xMinus = cloneArray( x );
				xCenter = cloneArray( x );
				
				xPlus.set( i, x.get( i ) + h );
				xMinus.set( i, x.get( i ) - h );
				
				hessian.set( i, j, ( f( xPlus ) - 2 * f( xCenter ) + f( xMinus ) ) / ( h * h ) );
			} else {
				// Off-diagonal elements use mixed partial derivatives
				xPlusPlus = cloneArray( x );
				xPlusMinus = cloneArray( x );
				xMinusPlus = cloneArray( x );
				xMinusMinus = cloneArray( x );
				
				xPlusPlus.set( i, x.get( i ) + h );
				xPlusPlus.set( j, x.get( j ) + h );
				
				xPlusMinus.set( i, x.get( i ) + h );
				xPlusMinus.set( j, x.get( j ) - h );
				
				xMinusPlus.set( i, x.get( i ) - h );
				xMinusPlus.set( j, x.get( j ) + h );
				
				xMinusMinus.set( i, x.get( i ) - h );
				xMinusMinus.set( j, x.get( j ) - h );
				
				// Mixed partial formula
				hessian.set( i, j, 
					( f( xPlusPlus ) - f( xPlusMinus ) - f( xMinusPlus ) + f( xMinusMinus ) ) / ( 4 * h * h )
				);
			}
		}
	}
	
	return hessian;
}

/**
* Calculates vector norm for convergence checking.
*
* @param {ndarray} v - Vector
* @returns {number} L2 norm of the vector
*/
function vectorNorm( v ) {
	var sum = 0;
	for ( var i = 0; i < v.shape[0]; i++ ) {
		sum += v.get( i ) * v.get( i );
	}
	return sqrt( sum );
}

// Define a linear system solver using BLAS operations
function solve( A, b ) {

    var ddot = blas.base.ddot;      
    var daxpy = blas.base.daxpy;    
    var idamax = blas.base.idamax;  
    var dcopy = blas.base.dcopy;    

	// Solver for Ax = b
    // For a 2x2 system, we can use Cramer's rule
    var n = A.shape[0];
    var x = array2d( new Float64Array( n ), { 'shape': [n] } );
    
    if ( n === 2 ) {
        var det = A.get( 0, 0 ) * A.get( 1, 1 ) - A.get( 0, 1 ) * A.get( 1, 0 );
        if ( abs( det ) < 1e-10 ) {

            // Return a small step in the steepest descent direction
            for ( var i = 0; i < n; i++ ) {
                x.set( i, 0.01 * b.get( i ) );
            }
            return x;
        }
        
        x.set( 0, ( A.get( 1, 1 ) * b.get( 0 ) - A.get( 0, 1 ) * b.get( 1 ) ) / det );
        x.set( 1, ( A.get( 0, 0 ) * b.get( 1 ) - A.get( 1, 0 ) * b.get( 0 ) ) / det );
        return x;
    }
    
    // For larger matrices, use Gaussian elimination with partial pivoting
    var A_data = new Float64Array( n * n );
    var b_data = new Float64Array( n );
    var x_data = new Float64Array( n );
    
    // Copy A and b into flat arrays
    for ( var i = 0; i < n; i++ ) {
        b_data[i] = b.get( i );
        for ( var j = 0; j < n; j++ ) {
            A_data[i*n + j] = A.get( i, j );
        }
    }
    
    // Create arrays for each row of A
    var rows = [];
    for ( var i = 0; i < n; i++ ) {
        rows.push( new Float64Array( A_data.buffer, i*n*8, n ) );
    }
    
    // Arrays for pivoting
    var ipiv = new Int32Array( n );
    for ( var i = 0; i < n; i++ ) {
        ipiv[i] = i;
    }
    
    // Gaussian elimination with partial pivoting
    for ( var k = 0; k < n-1; k++ ) {
        // Find pivot using idamax
        var col_k = new Float64Array( n-k );
        for ( var i = k; i < n; i++ ) {
            col_k[i-k] = A_data[i*n + k];
        }
        
        // Get index of max value (relative to the subarray)
        var max_idx = idamax( n-k, col_k, 1 );
        var p = k + max_idx; // Adjust to actual row index
        
        // Swap rows if necessary
        if ( p != k ) {
            // Swap rows in A
            var temp_row = new Float64Array( n );
            dcopy( n, rows[k], 1, temp_row, 1 );  // temp = rows[k]
            dcopy( n, rows[p], 1, rows[k], 1 );   // rows[k] = rows[p]
            dcopy( n, temp_row, 1, rows[p], 1 );  // rows[p] = temp
            
            // Swap elements in b
            var temp = b_data[k];
            b_data[k] = b_data[p];
            b_data[p] = temp;
            
            // Track pivoting
            temp = ipiv[k];
            ipiv[k] = ipiv[p];
            ipiv[p] = temp;
        }
        
        // Check for near zero pivot
        if ( abs( A_data[k*n + k] ) < 1e-10 ) {
            // Return a small step in the gradient direction as fallback
            for ( var i = 0; i < n; i++ ) {
                x.set( i, 0.01 * b.get( i ) );
            }
            return x;
        }
        
        // Eliminate below current row using daxpy
        for ( var i = k+1; i < n; i++ ) {
            var factor = -A_data[i*n + k] / A_data[k*n + k];
            
            // Row operation rows[i] = rows[i] + factor * rows[k]
            daxpy( n-k-1, factor, rows[k].subarray(k+1), 1, rows[i].subarray(k+1), 1 );
            
            // Update the eliminated element separately (not part of daxpy above)
            A_data[i*n + k] = 0;
            
            // Update b  b[i] += factor * b[k]
            b_data[i] += factor * b_data[k];
        }
    }
    
    // Back substitution
    for ( var i = n-1; i >= 0; i-- ) {
        var sum = b_data[i];
        
        if ( i < n-1 ) {
            // Use ddot to compute the dot product of row i and the solution so far
            sum -= ddot( n-i-1, rows[i].subarray(i+1), 1, x_data.subarray(i+1), 1 );
        }
        
        x_data[i] = sum / A_data[i*n + i];
    }
    
    // Copy solution to ndarray
    for ( var i = 0; i < n; i++ ) {
        x.set( i, x_data[i] );
    }
    
    return x;
}

/**
* Newton's method.
*
* @param {Function} f - The objective function f(x)
* @param {ndarray} initialX - Initial point
* @param {Object} [options] - Optimization options
* @param {number} [options.tolerance=1e-6] - Convergence tolerance
* @param {number} [options.maxIterations=100] - Maximum number of iterations
* @param {boolean} [options.useLineSearch=true] - Whether to use line search
* @param {number} [options.stepSize=1.0] - Initial step size for line search
* @returns {Object} Object containing solution, function value, iteration count, and convergence status
*/
function newtonsMethod( f, initialX, options ) {
	// Set defaults for options
	options = options || {};
	var tolerance = options.tolerance !== undefined ? options.tolerance : 1e-6;
	var maxIterations = options.maxIterations !== undefined ? options.maxIterations : 100;
	var useLineSearch = options.useLineSearch !== undefined ? options.useLineSearch : true;
	var stepSize = options.stepSize !== undefined ? options.stepSize : 1.0;
	
	// Initialize variables
	var x = cloneArray( initialX );
	var fx = f( x );
	var iterations = 0;
	var converged = false;
	
	// Store optimization path for visualization
	var path = [cloneArray( x )];
	var fValues = [fx];
	
	// Main optimization loop
	var grad, hessian, gradNorm, step;
	
	while ( iterations < maxIterations && !converged ) {
		// Compute gradient and Hessian
		grad = numericalGradient( f, x );
		hessian = numericalHessian( f, x );
		
		// Check grad norm for convergence
		gradNorm = vectorNorm( grad );
		if ( gradNorm < tolerance ) {
			converged = true;
			break;
		}
		
		// Negative of gradient to get descent direction
		for ( var i = 0; i < grad.shape[0]; i++ ) {
			grad.set( i, -grad.get( i ) );
		}
		
		// Solve linear system Hessian * step = -gradient
		step = solve( hessian, grad );
		
		// Apply step with or without line search
		if ( useLineSearch ) {
			// Simple backtracking line search
			var alpha = stepSize;
			var newX = cloneArray( x );
			for ( var i = 0; i < x.shape[0]; i++ ) {
				newX.set( i, x.get( i ) + alpha * step.get( i ) );
			}
			var newFx = f( newX );
			
			// Backtracking parameters
			var c = 0.5;  // Decrease factor
			var rho = 0.5;  // Sufficient decrease parameter
			
			// Backtrack until a sufficient decrease is found
			while ( newFx > fx - rho * alpha * gradNorm * vectorNorm( step ) && alpha > 1e-10 ) {
				alpha *= c;
				for ( var i = 0; i < x.shape[0]; i++ ) {
					newX.set( i, x.get( i ) + alpha * step.get( i ) );
				}
				newFx = f( newX );
			}
			
			x = newX;
			fx = newFx;
		} else {
			// Full Newton step
			for ( var i = 0; i < x.shape[0]; i++ ) {
				x.set( i, x.get( i ) + step.get( i ) );
			}
			fx = f( x );
		}
		
		// Store current point in optimization path
		path.push( cloneArray( x ) );
		fValues.push( fx );
		
		iterations++;
	}
	
	// Return the solution and information
	return {
		'solution': x,
		'value': fx,
		'iterations': iterations,
		'converged': converged,
		'path': path,
		'fValues': fValues
	};
}

/**
* Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
*
* @param {ndarray} x - Point [x,y]
* @returns {number} Function value
*/
function rosenbrock( x ) {
	var x1 = x.get( 0 );
	var x2 = x.get( 1 );
	return pow( 1 - x1, 2 ) + 100 * pow( x2 - x1 * x1, 2 );
}

/**
* Himmelblau function: f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2
*
* @param {ndarray} x - Point [x,y]
* @returns {number} Function value
*/
function himmelblau( x ) {
	var x1 = x.get( 0 );
	var x2 = x.get( 1 );
	return pow( x1 * x1 + x2 - 11, 2 ) + pow( x1 + x2 * x2 - 7, 2 );
}

/**
* Shows an example of Newton's method.
*
* @returns {Object} Optimization results
*/
function runExample() {
	// Create an init guess
	var initialX = array2d( new Float64Array( [0, 0] ), {
		'shape': [2]
	} );
	
	// Run Newtons method on the Rosenbrock function
	console.log( 'Optimizing Rosenbrock function...' );
	var result = newtonsMethod( rosenbrock, initialX, {
		'tolerance': 1e-6,
		'maxIterations': 100,
		'useLineSearch': true
	} );
	
	// Display results
	console.log( 'Solution:', [result.solution.get( 0 ), result.solution.get( 1 )] );
	console.log( 'Function value:', result.value );
	console.log( 'Iterations:', result.iterations );
	console.log( 'Converged:', result.converged );
	
	return result;
}

// Export API methods
module.exports = {
	newtonsMethod: newtonsMethod,
	numericalGradient: numericalGradient,
	numericalHessian: numericalHessian,
	rosenbrock: rosenbrock,
	himmelblau: himmelblau,
	runExample: runExample
};

// If run directly, execute the example
if ( require.main === module ) {
	runExample();
}
