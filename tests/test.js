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


var tape = require( 'tape' );
var newton = require( '../lib/main.js' );
var ndarray = require( '@stdlib/ndarray' );
var abs = require( '@stdlib/math/base/special/abs' );
var exp = require( '@stdlib/math/base/special/exp' );
var pow = require( '@stdlib/math/base/special/pow' );
var Float64Array = require( '@stdlib/array/float64' );

tape( 'main export is an object', function test( t ) {
	t.ok( true, 'file does not throw' );
	t.equal( typeof newton, 'object', 'main export is an object' );
	t.equal( typeof newton.newtonsMethod, 'function', 'newtonsMethod is a function' );
	t.equal( typeof newton.numericalGradient, 'function', 'numericalGradient is a function' );
	t.equal( typeof newton.numericalHessian, 'function', 'numericalHessian is a function' );
	t.end();
} );

tape( 'newtonsMethod optimizes a simple quadratic function', function test( t ) {
	var array2d = ndarray.array;
	var initialX = array2d( new Float64Array( [1, 1] ), {
		'shape': [2]
	} );
	
	// Simple quadratic function f(x) = x^2 + y^2, minimum at (0,0)
	function quadratic( x ) {
		return x.get( 0 ) * x.get( 0 ) + x.get( 1 ) * x.get( 1 );
	}
	
	var result = newton.newtonsMethod( quadratic, initialX );
	
	t.ok( result.converged, 'optimization converged' );
	t.ok( abs( result.solution.get( 0 ) ) < 1e-4, 'x coordinate close to minimum' );
	t.ok( abs( result.solution.get( 1 ) ) < 1e-4, 'y coordinate close to minimum' );
	t.ok( result.value < 1e-8, 'function value close to minimum' );
	t.end();
} );

tape( 'newtonsMethod optimizes the Rosenbrock function', function test( t ) {
	var array2d = ndarray.array;
	var initialX = array2d( new Float64Array( [0, 0] ), {
		'shape': [2]
	} );
	
	// Run with optimized parameters for this specific function
	var result = newton.newtonsMethod( newton.rosenbrock, initialX, {
		'tolerance': 1e-6,     
		'maxIterations': 5000, 
		'useLineSearch': true,
		'stepSize': 0.1        
	} );
	
	t.ok( result.value < 0.5, 'function value decreased from starting point' );
	
	// Verify we're in the correct basin
	var x = result.solution.get( 0 );
	var y = result.solution.get( 1 );
	t.ok( result.converged, 'optimization converged' );
	t.ok( x > 0.9 && x < 1.1, 'x coordinate in correct region' );
	t.ok( y > 0.9 && y < 1.1, 'y coordinate in correct region' );
	t.ok( result.value >= 0, 'function value physically meaningful' );
	
	t.end();
} );

tape('newtonsMethod optimizes the Himmelblau function', function test(t) {
    var array2d = ndarray.array;
    var initialX = array2d(new Float64Array([2, 3]), {
        'shape': [2]
    });

    // Run with optimized parameters for this specific function
    var result = newton.newtonsMethod(newton.himmelblau, initialX, {
        'tolerance': 1e-6,
        'maxIterations': 5000,
        'useLineSearch': true,
        'stepSize': 0.1
    });

    t.ok(result.value < 0.5, 'function value decreased from starting point');
    t.ok(result.converged, 'optimization converged');
    t.ok(result.value >= 0, 'function value physically meaningful');

    t.end();
});


tape( 'newtonsMethod optimizes a bowl function with multiple starting points', function test( t ) {
	// Bowl function f(x,y) = (x-3)^2 + (y+2)^2, minimum at (3,-2)
	function bowl( x ) {
		return pow( x.get( 0 ) - 3, 2 ) + pow( x.get( 1 ) + 2, 2 );
	}
	
	var array2d = ndarray.array;
	var startingPoints = [
		[0, 0],    // Origin
		[10, 10],  // Far away
		[3, 0],    // Correct x, wrong y
		[0, -2]    // Wrong x, correct y
	];
	
	startingPoints.forEach( function( point, i ) {
		var initialX = array2d( new Float64Array( point ), {
			'shape': [2]
		} );
		
		var result = newton.newtonsMethod( bowl, initialX );
		
		t.ok( result.converged, 'optimization from point ' + i + ' converged' );
		t.ok( abs( result.solution.get( 0 ) - 3.0 ) < 0.01, 'x coordinate from point ' + i + ' close to minimum' );
		t.ok( abs( result.solution.get( 1 ) + 2.0 ) < 0.01, 'y coordinate from point ' + i + ' close to minimum' );
	} );
	
	t.end();
} );

tape( 'newtonsMethod handles non-quadratic functions', function test( t ) {
	// Exponential function with known minimum at (0,0)
	function expFunction( x ) {
		return exp( x.get( 0 ) * x.get( 0 ) + x.get( 1 ) * x.get( 1 ) );
	}
	
	var array2d = ndarray.array;
	var initialX = array2d( new Float64Array( [0.5, 0.5] ), {
		'shape': [2]
	} );
	
	var result = newton.newtonsMethod( expFunction, initialX, {
		'maxIterations': 500
	} );
	
	t.ok( result.converged, 'optimization converged' );
	t.ok( abs( result.solution.get( 0 ) ) < 0.1, 'x coordinate close to minimum' );
	t.ok( abs( result.solution.get( 1 ) ) < 0.1, 'y coordinate close to minimum' );
	
	t.end();
} );

tape( 'numericalGradient correctly computes gradient', function test( t ) {
	var array2d = ndarray.array;
	var point = array2d( new Float64Array( [3, 2] ), {
		'shape': [2]
	} );
	
	// f(x,y) = x^2 + 2y^2, gradient at (3,2) is (6,8)
	function testFunction( x ) {
		return x.get( 0 ) * x.get( 0 ) + 2 * x.get( 1 ) * x.get( 1 );
	}
	
	var gradient = newton.numericalGradient( testFunction, point );
	
	t.ok( abs( gradient.get( 0 ) - 6.0 ) < 1e-4, 'x-gradient correct' );
	t.ok( abs( gradient.get( 1 ) - 8.0 ) < 1e-4, 'y-gradient correct' );
	t.end();
} );

tape( 'numericalHessian correctly computes Hessian matrix', function test( t ) {
	var array2d = ndarray.array;
	var point = array2d( new Float64Array( [1, 1] ), {
		'shape': [2]
	} );
	
	// f(x,y) = x^2 + xy + 2y^2, Hessian is [[2,1],[1,4]]
	function testFunction( x ) {
		return x.get( 0 ) * x.get( 0 ) + x.get( 0 ) * x.get( 1 ) + 2 * x.get( 1 ) * x.get( 1 );
	}
	
	var hessian = newton.numericalHessian( testFunction, point );
	
	t.ok( abs( hessian.get( 0, 0 ) - 2.0 ) < 0.1, 'H[0,0] correct' );
	t.ok( abs( hessian.get( 0, 1 ) - 1.0 ) < 0.1, 'H[0,1] correct' );
	t.ok( abs( hessian.get( 1, 0 ) - 1.0 ) < 0.1, 'H[1,0] correct' );
	t.ok( abs( hessian.get( 1, 1 ) - 4.0 ) < 0.1, 'H[1,1] correct' );
	t.end();
} );
